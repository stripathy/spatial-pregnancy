"""
Cell type classification and benchmarking pipeline.

Runs multiple classification approaches on MERFISH and Slide-tags datasets,
then benchmarks each against Keon's CAST annotations.

Approaches:
  1. Direct correlation (snRNAseq centroids -> query)
  2. Two-pass (snRNAseq centroids -> exemplars -> re-classify)
  3. Self-referencing (Slide-tags centroids -> MERFISH, and vice versa)

Benchmarks:
  1. Agreement with CAST labels (class + subclass)
  2. Cell proportion correlation with reference
  3. Spatial coherence (neighbor purity)
  4. Marker gene expression validation

All intermediate results saved to disk as we go.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sparse
from scipy.spatial import cKDTree
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── paths ────────────────────────────────────────────────────────────────
WORKING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(WORKING_DIR)

# ── shared modules ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(WORKING_DIR, 'code'))
from modules.correlation import correlate, assign_labels, normalize_expr
from modules.gene_mapping import load_gene_mapping, get_gene_indices

CENTROIDS_PATH = 'output/data/snrnaseq_centroids.npz'
MERFISH_GENE_MAPPING = 'data/genename_mapping_merfish_panel.csv'
GENE_SYMBOL_TO_ENSEMBL = 'data/reference/gene_symbol_to_ensembl_mouse.json'

# Pre-load gene mapping once at module level so call sites don't repeat it
_GENE_MAPPING = None

def _get_gene_mapping():
    global _GENE_MAPPING
    if _GENE_MAPPING is None:
        _GENE_MAPPING = load_gene_mapping(MERFISH_GENE_MAPPING, GENE_SYMBOL_TO_ENSEMBL)
    return _GENE_MAPPING

DATASETS = {
    'slide_tags': 'data/GSE313279_slide_tags.h5ad',
    'merfish': 'data/merfish_clean/merfish_normalized/adata_query_merfish_pub.h5ad',
}

OUTPUT_DIR = 'output/classification'
FIGURE_DIR = 'output/figures/classification'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# ── Canonical marker genes for validation ────────────────────────────────
MARKERS = {
    # Glutamatergic
    'Slc17a7': '01 IT-ET Glut',
    'Slc17a6': '13 CNU-HYa Glut',
    # GABAergic
    'Gad1': 'GABAergic',
    'Gad2': 'GABAergic',
    'Slc32a1': 'GABAergic',
    # Astrocytes
    'Aqp4': '30 Astro-Epen',
    'Gfap': '30 Astro-Epen',
    # Oligodendrocytes
    'Mbp': '31 OPC-Oligo',
    'Mog': '31 OPC-Oligo',
    'Pdgfra': '31 OPC-Oligo',
    # Microglia
    'Ctss': '34 Immune',
    'C1qa': '34 Immune',
    'Cx3cr1': '34 Immune',
    # Vascular
    'Cldn5': '33 Vascular',
    'Flt1': '33 Vascular',
}


# ════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ════════════════════════════════════════════════════════════════════════

def log(msg):
    """Print with timestamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_centroids():
    """Load prebuilt snRNAseq centroids."""
    data = np.load(CENTROIDS_PATH, allow_pickle=True)
    return {
        'subc_centroids': data['subc_centroids'],
        'subc_names': list(data['subc_names']),
        'clas_centroids': data['clas_centroids'],
        'clas_names': list(data['clas_names']),
        'ref_ensembl_ids': list(data['ref_ensembl_ids']),
        'ref_symbols': list(data['ref_symbols']),
        'subc_to_class': json.loads(str(data['subc_to_class'])),
    }


def _build_int_gene_index(query_genes, ref_ensembl_ids):
    """Return (query_int_idx, ref_int_idx) integer position arrays for shared genes."""
    gene_mapping = _get_gene_mapping()
    gene_order, ref_idx_list = get_gene_indices(list(query_genes), ref_ensembl_ids, gene_mapping)
    query_int_idx = np.array([list(query_genes).index(g) for g in gene_order])
    return query_int_idx, np.array(ref_idx_list)


def assign_from_correlation(corr, type_names, parent_map=None):
    """Assign labels from a correlation matrix, optionally mapping to parent types.

    Thin wrapper around assign_labels() that adds parent_map support.
    Returns: labels, best_corr, margin, parent_labels (or None).
    """
    labels, best_corr, margin = assign_labels(corr, type_names)
    parent_labels = ([parent_map.get(l, 'Unknown') for l in labels]
                     if parent_map else None)
    return list(labels), best_corr, margin, parent_labels


# ════════════════════════════════════════════════════════════════════════
# CLASSIFICATION METHODS
# ════════════════════════════════════════════════════════════════════════

def method_direct_correlation(query_log2cpm, query_idx, ref_idx,
                               centroids, level='subclass'):
    """
    Method 1: Direct correlation against snRNAseq centroids.
    """
    if level == 'subclass':
        cent = centroids['subc_centroids'][:, ref_idx]
        names = centroids['subc_names']
    else:
        cent = centroids['clas_centroids'][:, ref_idx]
        names = centroids['clas_names']

    qmat = query_log2cpm[:, query_idx]
    cent_df = pd.DataFrame(cent, index=names)
    corr, _ = correlate(qmat, cent_df, verbose=False)
    labels, conf, margin, parent = assign_from_correlation(
        corr, names,
        centroids['subc_to_class'] if level == 'subclass' else None
    )
    return labels, conf, margin, parent, corr


def method_two_pass(query_log2cpm, query_idx, ref_idx, centroids,
                    n_exemplars=100, min_exemplars=5):
    """
    Method 2: Two-pass exemplar-based classification.
    Pass 1: Classify against snRNAseq centroids.
    Pass 2: Select top exemplars, build query-derived centroids, re-classify.
    """
    # Pass 1
    subc_cent = centroids['subc_centroids'][:, ref_idx]
    subc_names = centroids['subc_names']
    qmat = query_log2cpm[:, query_idx]

    subc_df = pd.DataFrame(subc_cent, index=subc_names)
    corr1, _ = correlate(qmat, subc_df, verbose=False)
    labels1, conf1, margin1, class1 = assign_from_correlation(
        corr1, subc_names, centroids['subc_to_class']
    )

    # Select exemplars
    exemplar_indices = []
    exemplar_labels = []
    for sc_idx, sc_name in enumerate(subc_names):
        cell_mask = np.array(labels1) == sc_name
        if cell_mask.sum() == 0:
            continue
        cell_indices = np.where(cell_mask)[0]
        cell_corrs = corr1[cell_indices, sc_idx]
        n_pick = min(n_exemplars, len(cell_indices))
        if n_pick < min_exemplars:
            continue
        top_idx = cell_indices[np.argsort(cell_corrs)[-n_pick:]]
        exemplar_indices.extend(top_idx)
        exemplar_labels.extend([sc_name] * len(top_idx))

    exemplar_indices = np.array(exemplar_indices)

    # Build query-derived centroids
    query_centroids = {}
    for sc in sorted(set(exemplar_labels)):
        mask = np.array(exemplar_labels) == sc
        idx = exemplar_indices[mask]
        query_centroids[sc] = qmat[idx].mean(axis=0)

    qc_names = list(query_centroids.keys())
    qc_mat = np.array([query_centroids[k] for k in qc_names])

    # Pass 2
    qc_df = pd.DataFrame(qc_mat, index=qc_names)
    corr2, _ = correlate(qmat, qc_df, verbose=False)
    labels2, conf2, margin2, class2 = assign_from_correlation(
        corr2, qc_names, centroids['subc_to_class']
    )

    return labels2, conf2, margin2, class2, len(exemplar_indices), len(qc_names)


def method_cross_modality(source_adata, target_log2cpm, target_query_idx,
                          target_ref_idx, centroids, n_exemplars=100):
    """
    Method 3: Use one modality's CAST labels to build centroids,
    classify the other modality.

    source_adata: annotated adata with CAST labels + expression
    target_log2cpm: target query matrix
    target_query_idx, target_ref_idx: gene indices for target
    """
    # Build centroids from source CAST labels
    source_labels = source_adata.obs['subclass'].astype(str).values
    ref_ensembl_ids = centroids['ref_ensembl_ids']

    # Get source gene indices
    source_query_idx, source_ref_idx = _build_int_gene_index(
        list(source_adata.var_names), ref_ensembl_ids
    )

    # Normalize source
    source_log2cpm = normalize_expr(source_adata)

    # Find common ref indices between source and target
    source_ref_set = set(source_ref_idx)
    target_ref_set = set(target_ref_idx)
    common_ref = sorted(source_ref_set & target_ref_set)

    # Build source index -> common index mapping
    source_ref_to_qi = {ri: qi for qi, ri in zip(source_query_idx, source_ref_idx)}
    target_ref_to_qi = {ri: qi for qi, ri in zip(target_query_idx, target_ref_idx)}

    source_common_qi = [source_ref_to_qi[ri] for ri in common_ref]
    target_common_qi = [target_ref_to_qi[ri] for ri in common_ref]

    log(f"  Cross-modality: {len(common_ref)} common genes")

    # Build centroids from source
    source_X = source_log2cpm[:, source_common_qi]
    unique_sc = sorted(set(source_labels))
    src_centroids = {}
    for sc in unique_sc:
        mask = source_labels == sc
        if mask.sum() >= 20:
            src_centroids[sc] = source_X[mask].mean(axis=0)

    sc_names = list(src_centroids.keys())
    sc_mat = np.array([src_centroids[k] for k in sc_names])

    # Classify target
    target_X = target_log2cpm[:, target_common_qi]
    sc_df = pd.DataFrame(sc_mat, index=sc_names)
    corr, _ = correlate(target_X, sc_df, verbose=False)
    labels, conf, margin, class_labels = assign_from_correlation(
        corr, sc_names, centroids['subc_to_class']
    )

    return labels, conf, margin, class_labels


# ════════════════════════════════════════════════════════════════════════
# BENCHMARKING
# ════════════════════════════════════════════════════════════════════════

def benchmark_agreement(cast_labels, pred_labels, level_name):
    """Compute agreement between CAST and predicted labels."""
    cast = np.array(cast_labels, dtype=str)
    pred = np.array(pred_labels, dtype=str)
    overall = (cast == pred).mean()

    # Per-type breakdown
    per_type = {}
    for t in sorted(set(cast)):
        mask = cast == t
        n = mask.sum()
        if n >= 20:
            agree = (cast[mask] == pred[mask]).mean()
            per_type[t] = {'agreement': agree, 'n_cells': n}

    return overall, per_type


def benchmark_proportions(cast_labels, pred_labels, ref_proportions=None):
    """
    Compare cell type proportions between CAST and predicted labels.
    Returns Pearson and Spearman correlations.
    """
    cast_counts = pd.Series(cast_labels).value_counts(normalize=True)
    pred_counts = pd.Series(pred_labels).value_counts(normalize=True)

    # Align on common types
    common = sorted(set(cast_counts.index) & set(pred_counts.index))
    if len(common) < 3:
        return np.nan, np.nan, None

    cast_props = cast_counts.reindex(common, fill_value=0)
    pred_props = pred_counts.reindex(common, fill_value=0)

    from scipy.stats import pearsonr, spearmanr
    # Log-scale correlation (better for proportions spanning orders of magnitude)
    log_cast = np.log10(cast_props.values + 1e-6)
    log_pred = np.log10(pred_props.values + 1e-6)
    r_pearson, _ = pearsonr(log_cast, log_pred)
    r_spearman, _ = spearmanr(cast_props.values, pred_props.values)

    return r_pearson, r_spearman, pd.DataFrame({
        'cast_prop': cast_props, 'pred_prop': pred_props
    })


def benchmark_spatial_coherence(obs_df, label_col, x_col='x', y_col='y',
                                 k=15):
    """
    Spatial neighbor purity: for each cell, what fraction of its k nearest
    spatial neighbors share the same label?
    Higher = more spatially coherent annotations.
    """
    coords = obs_df[[x_col, y_col]].values
    labels = obs_df[label_col].values

    # Build per-sample KD trees
    samples = obs_df['sample'].unique() if 'sample' in obs_df.columns else ['all']
    purities = []

    for sample in samples:
        if sample == 'all':
            mask = np.ones(len(obs_df), dtype=bool)
        else:
            mask = obs_df['sample'].values == sample

        if mask.sum() < k + 1:
            continue

        sample_coords = coords[mask]
        sample_labels = labels[mask]

        tree = cKDTree(sample_coords)
        _, nn_idx = tree.query(sample_coords, k=k+1)  # +1 for self

        # Fraction of neighbors with same label (excluding self)
        for i in range(mask.sum()):
            neighbors = nn_idx[i, 1:]  # exclude self
            same = (sample_labels[neighbors] == sample_labels[i]).mean()
            purities.append(same)

    return np.mean(purities) if purities else 0.0


def benchmark_marker_expression(adata, label_col, markers=MARKERS):
    """
    Check if canonical marker genes are enriched in expected cell types.
    Returns fraction of markers correctly enriched.
    """
    available_markers = {g: t for g, t in markers.items()
                        if g in adata.var_names}
    if not available_markers:
        return 0.0, {}

    results = {}
    for gene, expected_class in available_markers.items():
        gene_idx = list(adata.var_names).index(gene)
        if sparse.issparse(adata.X):
            expr = np.asarray(adata.X[:, gene_idx].todense()).ravel()
        else:
            expr = adata.X[:, gene_idx].ravel()

        labels = adata.obs[label_col].astype(str).values

        # Mean expression per class
        class_means = {}
        for cl in sorted(set(labels)):
            mask = labels == cl
            if mask.sum() >= 10:
                class_means[cl] = expr[mask].mean()

        if not class_means:
            continue

        # Check if expected class has highest or near-highest expression
        sorted_classes = sorted(class_means.items(), key=lambda x: -x[1])
        top_class = sorted_classes[0][0]

        # For broad markers like "GABAergic", check if top class contains the keyword
        if expected_class == 'GABAergic':
            correct = 'GABA' in top_class
        else:
            correct = top_class == expected_class

        results[gene] = {
            'expected': expected_class,
            'top_class': top_class,
            'top_expr': sorted_classes[0][1],
            'correct': correct,
        }

    fraction_correct = sum(r['correct'] for r in results.values()) / len(results) \
        if results else 0.0
    return fraction_correct, results


# ════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════

def run_dataset(dataset_name, centroids):
    """Run all classification methods and benchmarks for one dataset."""
    log(f"\n{'='*70}")
    log(f"Processing: {dataset_name}")
    log(f"{'='*70}")

    # ── Load data ────────────────────────────────────────────────────
    log(f"Loading {dataset_name}...")
    adata = ad.read_h5ad(DATASETS[dataset_name])
    log(f"  {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")

    # ── Gene mapping ─────────────────────────────────────────────────
    query_idx, ref_idx = _build_int_gene_index(
        list(adata.var_names),
        centroids['ref_ensembl_ids']
    )
    log(f"  {len(query_idx)} genes mapped to reference")

    # ── Normalize ────────────────────────────────────────────────────
    log(f"Normalizing to log-CPM...")
    query_log2cpm = normalize_expr(adata)
    log(f"  Range: [{query_log2cpm.min():.2f}, {query_log2cpm.max():.2f}]")

    # ── CAST labels (ground truth for comparison) ────────────────────
    cast_class = adata.obs['class'].astype(str).values
    cast_subclass = adata.obs['subclass'].astype(str).values

    results = {}

    # ════════════════════════════════════════════════════════════════
    # Method 1: Direct correlation (snRNAseq centroids)
    # ════════════════════════════════════════════════════════════════
    log(f"\nMethod 1: Direct correlation (snRNAseq centroids)...")
    t1 = time.time()
    m1_sub, m1_sub_conf, m1_sub_margin, m1_class, m1_corr = \
        method_direct_correlation(query_log2cpm, query_idx, ref_idx,
                                   centroids, level='subclass')
    log(f"  Done in {time.time()-t1:.1f}s")

    adata.obs['m1_subclass'] = m1_sub
    adata.obs['m1_class'] = m1_class
    adata.obs['m1_subclass_conf'] = m1_sub_conf
    adata.obs['m1_subclass_margin'] = m1_sub_margin

    # Save intermediate
    pd.DataFrame({
        'cell': adata.obs.index,
        'm1_subclass': m1_sub,
        'm1_class': m1_class,
        'm1_conf': m1_sub_conf,
        'm1_margin': m1_sub_margin,
    }).to_csv(f'{OUTPUT_DIR}/{dataset_name}_m1_direct.csv', index=False)
    log(f"  Saved {dataset_name}_m1_direct.csv")

    # ════════════════════════════════════════════════════════════════
    # Method 2: Two-pass exemplar-based
    # ════════════════════════════════════════════════════════════════
    log(f"\nMethod 2: Two-pass exemplar-based...")
    t2 = time.time()
    m2_sub, m2_sub_conf, m2_sub_margin, m2_class, n_exemplars, n_centroids = \
        method_two_pass(query_log2cpm, query_idx, ref_idx, centroids,
                        n_exemplars=100)
    log(f"  Done in {time.time()-t2:.1f}s "
        f"({n_exemplars} exemplars, {n_centroids} centroids)")

    adata.obs['m2_subclass'] = m2_sub
    adata.obs['m2_class'] = m2_class
    adata.obs['m2_subclass_conf'] = m2_sub_conf
    adata.obs['m2_subclass_margin'] = m2_sub_margin

    pd.DataFrame({
        'cell': adata.obs.index,
        'm2_subclass': m2_sub,
        'm2_class': m2_class,
        'm2_conf': m2_sub_conf,
        'm2_margin': m2_sub_margin,
    }).to_csv(f'{OUTPUT_DIR}/{dataset_name}_m2_twopass.csv', index=False)
    log(f"  Saved {dataset_name}_m2_twopass.csv")

    # ════════════════════════════════════════════════════════════════
    # Method 3: Two-pass with MORE exemplars (200) and stricter min
    # ════════════════════════════════════════════════════════════════
    log(f"\nMethod 3: Two-pass (200 exemplars, min 10)...")
    t3 = time.time()
    m3_sub, m3_sub_conf, m3_sub_margin, m3_class, n_ex3, n_ct3 = \
        method_two_pass(query_log2cpm, query_idx, ref_idx, centroids,
                        n_exemplars=200, min_exemplars=10)
    log(f"  Done in {time.time()-t3:.1f}s "
        f"({n_ex3} exemplars, {n_ct3} centroids)")

    adata.obs['m3_subclass'] = m3_sub
    adata.obs['m3_class'] = m3_class
    adata.obs['m3_subclass_conf'] = m3_sub_conf

    pd.DataFrame({
        'cell': adata.obs.index,
        'm3_subclass': m3_sub,
        'm3_class': m3_class,
        'm3_conf': m3_sub_conf,
    }).to_csv(f'{OUTPUT_DIR}/{dataset_name}_m3_twopass200.csv', index=False)
    log(f"  Saved {dataset_name}_m3_twopass200.csv")

    # ════════════════════════════════════════════════════════════════
    # BENCHMARKS
    # ════════════════════════════════════════════════════════════════
    log(f"\n{'─'*70}")
    log(f"Running benchmarks for {dataset_name}...")
    log(f"{'─'*70}")

    methods = {
        'CAST (original)': ('class', 'subclass'),
        'M1: Direct corr': ('m1_class', 'm1_subclass'),
        'M2: Two-pass (100)': ('m2_class', 'm2_subclass'),
        'M3: Two-pass (200)': ('m3_class', 'm3_subclass'),
    }

    benchmark_rows = []

    for method_name, (class_col, sub_col) in methods.items():
        log(f"\n  Benchmarking: {method_name}")
        row = {'method': method_name, 'dataset': dataset_name}

        # 1. Agreement with CAST
        if method_name != 'CAST (original)':
            cls_agree, cls_per_type = benchmark_agreement(
                cast_class, adata.obs[class_col].astype(str).values, 'class')
            sub_agree, sub_per_type = benchmark_agreement(
                cast_subclass, adata.obs[sub_col].astype(str).values, 'subclass')
            row['class_agreement'] = cls_agree
            row['subclass_agreement'] = sub_agree
            log(f"    CAST agreement: class={100*cls_agree:.1f}%, "
                f"subclass={100*sub_agree:.1f}%")

            # Save per-type agreement
            pd.DataFrame(cls_per_type).T.to_csv(
                f'{OUTPUT_DIR}/{dataset_name}_{sub_col}_per_class_agreement.csv')
        else:
            row['class_agreement'] = 1.0
            row['subclass_agreement'] = 1.0

        # 2. Proportion correlation with reference
        # Use the reference subclass proportions from Zeng
        pred_sub = adata.obs[sub_col].astype(str).values
        r_pearson, r_spearman, prop_df = benchmark_proportions(
            cast_subclass, pred_sub)
        row['prop_pearson_r'] = r_pearson
        row['prop_spearman_r'] = r_spearman
        log(f"    Proportion correlation: Pearson r={r_pearson:.3f}, "
            f"Spearman r={r_spearman:.3f}")

        if prop_df is not None:
            prop_df.to_csv(
                f'{OUTPUT_DIR}/{dataset_name}_{sub_col}_proportions.csv')

        # 3. Spatial coherence
        if 'x' in adata.obs.columns and 'y' in adata.obs.columns:
            # Class-level spatial coherence
            obs_for_spatial = adata.obs.copy()
            obs_for_spatial['_label'] = obs_for_spatial[class_col].astype(str)
            cls_purity = benchmark_spatial_coherence(
                obs_for_spatial, '_label', k=15)
            row['spatial_class_purity'] = cls_purity

            # Subclass-level
            obs_for_spatial['_label'] = obs_for_spatial[sub_col].astype(str)
            sub_purity = benchmark_spatial_coherence(
                obs_for_spatial, '_label', k=15)
            row['spatial_subclass_purity'] = sub_purity
            log(f"    Spatial purity: class={cls_purity:.3f}, "
                f"subclass={sub_purity:.3f}")

        # 4. Marker gene validation
        marker_frac, marker_results = benchmark_marker_expression(
            adata, class_col)
        row['marker_fraction_correct'] = marker_frac
        log(f"    Marker validation: {100*marker_frac:.1f}% correct "
            f"({len(marker_results)} markers)")

        benchmark_rows.append(row)

    # ── Save benchmark summary ───────────────────────────────────────
    bench_df = pd.DataFrame(benchmark_rows)
    bench_df.to_csv(f'{OUTPUT_DIR}/{dataset_name}_benchmark_summary.csv',
                    index=False)
    log(f"\nSaved benchmark summary to {dataset_name}_benchmark_summary.csv")

    # ── Print summary table ──────────────────────────────────────────
    log(f"\n{'='*70}")
    log(f"BENCHMARK SUMMARY: {dataset_name}")
    log(f"{'='*70}")
    print(bench_df.to_string(index=False, float_format='%.3f'))

    # ── Generate figures ─────────────────────────────────────────────
    generate_figures(adata, dataset_name, methods, centroids)

    # ── Save annotated adata ─────────────────────────────────────────
    output_h5ad = f'{OUTPUT_DIR}/{dataset_name}_classified.h5ad'
    adata.write(output_h5ad)
    log(f"Saved annotated data to {output_h5ad}")

    return bench_df


def generate_figures(adata, dataset_name, methods, centroids):
    """Generate benchmark figures."""
    log(f"\nGenerating figures for {dataset_name}...")

    # ── 1. Spatial plots for each method ─────────────────────────────
    if 'x' in adata.obs.columns and 'y' in adata.obs.columns:
        method_cols = {
            'CAST': 'class',
            'M1_Direct': 'm1_class',
            'M2_TwoPass100': 'm2_class',
            'M3_TwoPass200': 'm3_class',
        }

        # Class-level spatial comparison
        fig, axes = plt.subplots(1, len(method_cols), figsize=(7*len(method_cols), 7))
        if len(method_cols) == 1:
            axes = [axes]

        for ax, (mname, col) in zip(axes, method_cols.items()):
            # Color by class
            labels = adata.obs[col].astype(str).values
            unique_labels = sorted(set(labels))
            cmap = plt.cm.tab20
            colors = {l: cmap(i/len(unique_labels)) for i, l in enumerate(unique_labels)}

            for sample in adata.obs['sample'].unique():
                smask = adata.obs['sample'].values == sample
                x = adata.obs['x'].values[smask]
                y = adata.obs['y'].values[smask]
                c = [colors.get(l, (0.5, 0.5, 0.5, 0.3)) for l in labels[smask]]
                ax.scatter(x, y, c=c, s=0.1, rasterized=True)
                break  # Just first sample for overview

            ax.set_title(f'{mname}', fontsize=16)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')

        plt.suptitle(f'{dataset_name} - Class annotations (first sample)',
                     fontsize=20)
        plt.tight_layout()
        plt.savefig(f'{FIGURE_DIR}/{dataset_name}_spatial_class_comparison.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        log(f"  Saved spatial class comparison figure")

    # ── 2. Confidence distributions ──────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (col, label) in zip(axes, [
        ('m1_subclass_conf', 'M1: Direct corr'),
        ('m2_subclass_conf', 'M2: Two-pass (100)'),
        ('m3_subclass_conf', 'M3: Two-pass (200)'),
    ]):
        if col in adata.obs.columns:
            ax.hist(adata.obs[col].values, bins=50, alpha=0.7,
                    edgecolor='black', linewidth=0.5)
            ax.set_xlabel('Correlation confidence', fontsize=14)
            ax.set_ylabel('Cells', fontsize=14)
            ax.set_title(f'{label}\nmedian={adata.obs[col].median():.3f}',
                        fontsize=14)
            ax.axvline(adata.obs[col].median(), color='red', linestyle='--')
    plt.suptitle(f'{dataset_name} - Subclass confidence distributions',
                 fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/{dataset_name}_confidence_distributions.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  Saved confidence distribution figure")

    # ── 3. Proportion scatter: CAST vs predicted ─────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (sub_col, label) in zip(axes, [
        ('m1_subclass', 'M1: Direct'),
        ('m2_subclass', 'M2: Two-pass (100)'),
        ('m3_subclass', 'M3: Two-pass (200)'),
    ]):
        cast_props = pd.Series(adata.obs['subclass'].astype(str)).value_counts(
            normalize=True)
        pred_props = pd.Series(adata.obs[sub_col].astype(str)).value_counts(
            normalize=True)
        common = sorted(set(cast_props.index) & set(pred_props.index))
        if len(common) < 3:
            continue

        x = np.log10(cast_props.reindex(common, fill_value=1e-6).values + 1e-6)
        y = np.log10(pred_props.reindex(common, fill_value=1e-6).values + 1e-6)

        ax.scatter(x, y, s=30, alpha=0.7)
        ax.plot([-5, 0], [-5, 0], 'k--', alpha=0.3)
        ax.set_xlabel('log10(CAST proportion)', fontsize=14)
        ax.set_ylabel('log10(Predicted proportion)', fontsize=14)

        from scipy.stats import pearsonr
        r, _ = pearsonr(x, y)
        ax.set_title(f'{label}\nr={r:.3f}', fontsize=14)

    plt.suptitle(f'{dataset_name} - Subclass proportions (CAST vs Predicted)',
                 fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/{dataset_name}_proportion_scatter.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  Saved proportion scatter figure")

    # ── 4. Per-class agreement heatmap ───────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    method_data = {}
    cast_class = adata.obs['class'].astype(str).values
    for sub_col, label in [
        ('m1_class', 'M1: Direct'),
        ('m2_class', 'M2: TwoPass100'),
        ('m3_class', 'M3: TwoPass200'),
    ]:
        pred = adata.obs[sub_col].astype(str).values
        per_type = {}
        for t in sorted(set(cast_class)):
            mask = cast_class == t
            if mask.sum() >= 20:
                per_type[t] = (cast_class[mask] == pred[mask]).mean()
        method_data[label] = per_type

    if method_data:
        all_types = sorted(set().union(*[set(v.keys()) for v in method_data.values()]))
        heatmap_data = pd.DataFrame(method_data, index=all_types).T
        im = ax.imshow(heatmap_data.values, cmap='RdYlGn', vmin=0, vmax=1,
                       aspect='auto')
        ax.set_xticks(range(len(all_types)))
        ax.set_xticklabels(all_types, rotation=90, fontsize=10)
        ax.set_yticks(range(len(method_data)))
        ax.set_yticklabels(list(method_data.keys()), fontsize=12)
        plt.colorbar(im, ax=ax, label='Agreement with CAST')
        ax.set_title(f'{dataset_name} - Per-class agreement with CAST',
                     fontsize=16)
        # Add text annotations
        for i in range(heatmap_data.shape[0]):
            for j in range(heatmap_data.shape[1]):
                val = heatmap_data.iloc[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{100*val:.0f}', ha='center', va='center',
                            fontsize=8, color='black' if val > 0.5 else 'white')

    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/{dataset_name}_per_class_agreement.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  Saved per-class agreement heatmap")


def run_cross_modality(centroids, all_results):
    """Run cross-modality classification (Slide-tags -> MERFISH and vice versa)."""
    log(f"\n{'='*70}")
    log(f"Cross-modality classification")
    log(f"{'='*70}")

    # Load both datasets
    log("Loading Slide-tags...")
    st = ad.read_h5ad(DATASETS['slide_tags'])
    log("Loading MERFISH...")
    mf = ad.read_h5ad(DATASETS['merfish'])

    # Get gene indices for each
    st_qi, st_ri = _build_int_gene_index(list(st.var_names), centroids['ref_ensembl_ids'])
    mf_qi, mf_ri = _build_int_gene_index(list(mf.var_names), centroids['ref_ensembl_ids'])

    st_log2 = normalize_expr(st)
    mf_log2 = normalize_expr(mf)

    # ── Slide-tags -> MERFISH ────────────────────────────────────────
    log("\nSlide-tags CAST labels -> MERFISH classification...")
    m4_sub, m4_conf, m4_margin, m4_class = method_cross_modality(
        st, mf_log2, mf_qi, mf_ri, centroids)

    mf.obs['m4_subclass'] = m4_sub
    mf.obs['m4_class'] = m4_class
    mf.obs['m4_subclass_conf'] = m4_conf

    cast_class = mf.obs['class'].astype(str).values
    cast_sub = mf.obs['subclass'].astype(str).values
    cls_agree = (cast_class == np.array(m4_class)).mean()
    sub_agree = (cast_sub == np.array(m4_sub)).mean()
    log(f"  ST->MF: class={100*cls_agree:.1f}%, subclass={100*sub_agree:.1f}%")

    pd.DataFrame({
        'cell': mf.obs.index,
        'm4_subclass': m4_sub,
        'm4_class': m4_class,
        'm4_conf': m4_conf,
    }).to_csv(f'{OUTPUT_DIR}/merfish_m4_cross_from_slidetags.csv', index=False)

    # ── MERFISH -> Slide-tags ────────────────────────────────────────
    log("\nMERFISH CAST labels -> Slide-tags classification...")
    m5_sub, m5_conf, m5_margin, m5_class = method_cross_modality(
        mf, st_log2, st_qi, st_ri, centroids)

    st.obs['m5_subclass'] = m5_sub
    st.obs['m5_class'] = m5_class
    st.obs['m5_subclass_conf'] = m5_conf

    cast_class_st = st.obs['class'].astype(str).values
    cast_sub_st = st.obs['subclass'].astype(str).values
    cls_agree_st = (cast_class_st == np.array(m5_class)).mean()
    sub_agree_st = (cast_sub_st == np.array(m5_sub)).mean()
    log(f"  MF->ST: class={100*cls_agree_st:.1f}%, subclass={100*sub_agree_st:.1f}%")

    pd.DataFrame({
        'cell': st.obs.index,
        'm5_subclass': m5_sub,
        'm5_class': m5_class,
        'm5_conf': m5_conf,
    }).to_csv(f'{OUTPUT_DIR}/slide_tags_m5_cross_from_merfish.csv', index=False)

    return {
        'ST_to_MF': {'class': cls_agree, 'subclass': sub_agree},
        'MF_to_ST': {'class': cls_agree_st, 'subclass': sub_agree_st},
    }


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    log("Starting classification and benchmarking pipeline")
    log(f"Working directory: {os.getcwd()}")

    # Load centroids
    log("Loading snRNAseq centroids...")
    centroids = load_centroids()
    log(f"  {len(centroids['subc_names'])} subclasses, "
        f"{len(centroids['clas_names'])} classes, "
        f"{len(centroids['ref_ensembl_ids'])} genes")

    all_results = {}

    # Run each dataset
    for ds in ['slide_tags', 'merfish']:
        try:
            bench = run_dataset(ds, centroids)
            all_results[ds] = bench
        except Exception as e:
            log(f"ERROR processing {ds}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Cross-modality
    try:
        cross_results = run_cross_modality(centroids, all_results)
        log(f"\nCross-modality results:")
        for k, v in cross_results.items():
            log(f"  {k}: class={100*v['class']:.1f}%, subclass={100*v['subclass']:.1f}%")
    except Exception as e:
        log(f"ERROR in cross-modality: {e}")
        import traceback
        traceback.print_exc()

    # ── Final summary ────────────────────────────────────────────────
    log(f"\n{'='*70}")
    log(f"FINAL SUMMARY")
    log(f"{'='*70}")

    if all_results:
        combined = pd.concat(all_results.values(), ignore_index=True)
        combined.to_csv(f'{OUTPUT_DIR}/all_benchmarks_summary.csv', index=False)
        log(f"\nAll benchmarks:")
        print(combined.to_string(index=False, float_format='%.3f'))

    elapsed = time.time() - t_start
    log(f"\nTotal pipeline time: {elapsed/60:.1f} minutes")
    log("Done!")


if __name__ == '__main__':
    main()
