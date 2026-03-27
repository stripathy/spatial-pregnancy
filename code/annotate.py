#!/usr/bin/env python3
"""
Cell type annotation pipeline for spatial transcriptomics data.

Annotates query cells (MERFISH, Slide-tags, Xenium 5k) using region-restricted
snRNAseq centroids from the Allen Brain Cell Atlas (WMB taxonomy).

Methods:
  - Direct correlation: Pearson (MERFISH/Xenium) or Spearman (Slide-tags/genome-wide)
  - Hierarchical: class first, then subclass/supertype within class

Reference centroids are built from precomputed_stats_ABC_revision_230821.h5,
restricted to cell types present in 4 coronal Zeng MERFISH reference sections
(C57BL6J-638850.46-.49).

Usage:
    python code/annotate.py --input data/my_query.h5ad --output output/my_query_annotated.h5ad
    python code/annotate.py --input data/my_query.h5ad --output output/my_query_annotated.h5ad --method spearman
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
from scipy.stats import rankdata
from collections import defaultdict
import h5py

WORKING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKING_DIR, 'code'))
from modules.gene_mapping import load_gene_mapping, get_gene_indices
from modules.correlation import spatial_coherence as _spatial_coherence


# ── Core functions ────────────────────────────────────────────────────────

def standardize_rows(X):
    """Row-wise z-score standardization."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1
    return (X - mu) / sd


def rank_rows(X):
    """Row-wise rank transform."""
    return np.apply_along_axis(
        lambda x: rankdata(x, method='average'), 1, X
    ).astype(np.float32)


def correlate_classify(query_X, centroids_dict, n_genes):
    """Classify cells by max Pearson correlation to centroids.

    Returns: labels, confidences, margins, full correlation matrix
    """
    names = list(centroids_dict.keys())
    cmat = np.array(list(centroids_dict.values()))
    q_std = standardize_rows(query_X)
    c_std = standardize_rows(cmat)
    corr = (q_std @ c_std.T) / n_genes
    best_idx = np.argmax(corr, axis=1)
    best_corr = corr[np.arange(len(corr)), best_idx]
    if corr.shape[1] >= 2:
        sorted_corr = np.sort(corr, axis=1)
        margin = sorted_corr[:, -1] - sorted_corr[:, -2]
    else:
        margin = np.ones(len(corr), dtype=np.float32)
    labels = [names[i] for i in best_idx]
    return labels, best_corr, margin, corr


def build_taxonomy(stats_path):
    """Load taxonomy tree and build level mappings from precomputed stats."""
    f = h5py.File(stats_path, 'r')
    tree = json.loads(f['taxonomy_tree'][()])
    hierarchy = tree['hierarchy']
    nm = tree['name_mapper']
    c2r = json.loads(f['cluster_to_row'][()])
    sums = f['sum'][:]
    ref_gene_ensembl = json.loads(f['col_names'][()])
    f.close()

    # Build cluster -> class/subclass/supertype mappings
    clas_data = tree[hierarchy[0]]
    subc_data = tree[hierarchy[1]]

    clus_to_class, clus_to_subc, clus_to_supt = {}, {}, {}
    subc_to_class, supt_to_subc, supt_to_class = {}, {}, {}

    for clas_id, subc_ids in clas_data.items():
        cn = nm[hierarchy[0]][clas_id]['name']
        for subc_id in subc_ids:
            sn = nm[hierarchy[1]][subc_id]['name']
            subc_to_class[sn] = cn
            if subc_id in subc_data:
                for supt_id in subc_data[subc_id]:
                    spn = nm[hierarchy[2]][supt_id]['name']
                    supt_to_subc[spn] = sn
                    supt_to_class[spn] = cn
                    if supt_id in tree[hierarchy[2]]:
                        for clus_id in tree[hierarchy[2]][supt_id]:
                            clus_to_class[clus_id] = cn
                            clus_to_subc[clus_id] = sn
                            clus_to_supt[clus_id] = spn

    return {
        'c2r': c2r, 'sums': sums, 'ref_gene_ensembl': ref_gene_ensembl,
        'clus_to_class': clus_to_class, 'clus_to_subc': clus_to_subc,
        'clus_to_supt': clus_to_supt, 'subc_to_class': subc_to_class,
        'supt_to_subc': supt_to_subc, 'supt_to_class': supt_to_class,
    }


def build_restricted_centroids(taxonomy, valid_types, gene_indices, level='class'):
    """Build mean centroids restricted to valid types, for specified gene indices."""
    level_map = {
        'class': taxonomy['clus_to_class'],
        'subclass': taxonomy['clus_to_subc'],
        'supertype': taxonomy['clus_to_supt'],
    }[level]

    c2r = taxonomy['c2r']
    sums = taxonomy['sums']

    type_sums = defaultdict(lambda: np.zeros(len(gene_indices), dtype=np.float64))
    type_counts = defaultdict(int)

    for clus_id, row_idx in c2r.items():
        type_name = level_map.get(clus_id)
        if type_name and type_name in valid_types:
            type_sums[type_name] += sums[row_idx, gene_indices]
            type_counts[type_name] += 1

    return {k: type_sums[k] / type_counts[k] for k in type_sums if type_counts[k] >= 1}


# get_gene_indices and load_gene_mapping imported from modules.gene_mapping


def prepare_expression(adata, gene_order, method='pearson'):
    """Extract and normalize expression matrix for classification.

    method='pearson': log2(CPM+1) normalization (for targeted panels like MERFISH)
    method='spearman': rank transform (for genome-wide like Slide-tags)
    """
    X = adata[:, gene_order].X
    if sparse.issparse(X):
        X = X.toarray()
    X = X.astype(np.float64)

    if method == 'spearman':
        # Rank transform — robust to distributional differences
        return standardize_rows(rank_rows(X))
    else:
        # Log2 CPM normalization
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return np.log2(X / row_sums * 1e6 + 1).astype(np.float32)


def hierarchical_classify(qX, class_centroids, subc_centroids, supt_centroids,
                          subc_to_class, supt_to_class, n_genes, use_spearman=False):
    """Hierarchical classification: class -> subclass -> supertype within class.

    If use_spearman, centroids should already be rank-transformed.
    """
    n_cells = qX.shape[0]

    # Step 1: classify to class
    m_class, m_class_conf, _, _ = correlate_classify(qX, class_centroids, n_genes)

    # Step 2: within each class, classify to subclass and supertype
    m_subc = np.array(['Unknown'] * n_cells, dtype=object)
    m_subc_conf = np.zeros(n_cells, dtype=np.float32)
    m_supt = np.array(['Unknown'] * n_cells, dtype=object)
    m_supt_conf = np.zeros(n_cells, dtype=np.float32)

    for cls_name in set(m_class):
        cls_mask = np.array(m_class) == cls_name
        if cls_mask.sum() == 0:
            continue
        cell_X = qX[cls_mask]

        # Subclasses in this class
        cls_subs = {sc: c for sc, c in subc_centroids.items()
                    if subc_to_class.get(sc) == cls_name}
        if cls_subs:
            labels, confs, _, _ = correlate_classify(cell_X, cls_subs, n_genes)
            m_subc[cls_mask] = labels
            m_subc_conf[cls_mask] = confs

        # Supertypes in this class
        cls_supts = {st: c for st, c in supt_centroids.items()
                     if supt_to_class.get(st) == cls_name}
        if cls_supts:
            labels, confs, _, _ = correlate_classify(cell_X, cls_supts, n_genes)
            m_supt[cls_mask] = labels
            m_supt_conf[cls_mask] = confs

    return {
        'class': np.array(m_class),
        'class_conf': np.array(m_class_conf, dtype=np.float32),
        'subclass': m_subc,
        'subclass_conf': m_subc_conf,
        'supertype': m_supt,
        'supertype_conf': m_supt_conf,
    }


# spatial_coherence imported from modules.correlation as _spatial_coherence
# Call: _spatial_coherence(labels_array, coords, k=20).mean()


# ── Main pipeline ─────────────────────────────────────────────────────────

def annotate_dataset(input_path, output_path, stats_path, valid_types_path,
                     merfish_mapping_path, symbol_to_ensembl_path,
                     method='pearson', sample_col='sample', prefix='hier'):
    """Full annotation pipeline for one dataset.

    Args:
        input_path: path to query h5ad
        output_path: path for annotated h5ad output
        stats_path: path to precomputed_stats HDF5
        valid_types_path: path to Zeng reference h5ad (for valid type lists)
        merfish_mapping_path: path to MERFISH gene panel mapping CSV
        symbol_to_ensembl_path: path to general symbol->ensembl JSON
        method: 'pearson' or 'spearman'
        sample_col: column in obs with sample IDs
        prefix: column prefix for annotations (default 'hier')
    """
    t0 = time.time()

    print(f"Loading taxonomy from {stats_path}...")
    taxonomy = build_taxonomy(stats_path)

    # Get valid types from Zeng reference sections
    print(f"Loading valid types from {valid_types_path}...")
    ref = ad.read_h5ad(valid_types_path, backed='r')
    valid_classes = set(ref.obs['class'].unique())
    valid_subclasses = set(ref.obs['subclass'].unique())
    valid_supertypes = set(ref.obs['supertype'].unique())
    ref.file.close()
    print(f"  Valid: {len(valid_classes)} classes, {len(valid_subclasses)} subclasses, "
          f"{len(valid_supertypes)} supertypes")

    # Load query
    print(f"Loading query from {input_path}...")
    adata = ad.read_h5ad(input_path)
    print(f"  {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")

    # Build gene mapping
    gene_mapping = load_gene_mapping(
        merfish_mapping_path=merfish_mapping_path,
        symbol_to_ensembl_path=symbol_to_ensembl_path
    )

    gene_order, ref_indices = get_gene_indices(
        adata.var_names, taxonomy['ref_gene_ensembl'], gene_mapping
    )
    print(f"  Gene overlap: {len(gene_order)}/{adata.shape[1]}")

    # Build centroids
    print("Building region-restricted centroids...")
    class_c = build_restricted_centroids(taxonomy, valid_classes, ref_indices, 'class')
    subc_c = build_restricted_centroids(taxonomy, valid_subclasses, ref_indices, 'subclass')
    supt_c = build_restricted_centroids(taxonomy, valid_supertypes, ref_indices, 'supertype')
    print(f"  {len(class_c)} classes, {len(subc_c)} subclasses, {len(supt_c)} supertypes")

    # Process each sample
    samples = sorted(adata.obs[sample_col].unique())
    print(f"Processing {len(samples)} samples with method={method}...")

    use_spearman = method == 'spearman'

    # Prepare centroids for spearman if needed
    if use_spearman:
        class_c = {k: standardize_rows(rank_rows(v.reshape(1, -1)))[0] for k, v in class_c.items()}
        subc_c = {k: standardize_rows(rank_rows(v.reshape(1, -1)))[0] for k, v in subc_c.items()}
        supt_c = {k: standardize_rows(rank_rows(v.reshape(1, -1)))[0] for k, v in supt_c.items()}

    n_genes = len(gene_order)
    all_results = []

    for sample in samples:
        query = adata[adata.obs[sample_col] == sample]
        print(f"  {sample}: {query.shape[0]:,} cells", end='')

        qX = prepare_expression(query, gene_order, method=method)

        result = hierarchical_classify(
            qX, class_c, subc_c, supt_c,
            taxonomy['subc_to_class'], taxonomy['supt_to_class'],
            n_genes, use_spearman=use_spearman
        )

        # Store results
        idx = adata.obs[sample_col] == sample
        for level in ['class', 'subclass', 'supertype']:
            adata.obs.loc[idx, f'{prefix}_{level}'] = result[level]
            adata.obs.loc[idx, f'{prefix}_{level}_conf'] = result[f'{level}_conf']

        # Quick benchmark if CAST labels exist
        if 'class' in adata.obs.columns and 'subclass' in adata.obs.columns:
            cast_c = query.obs['class'].astype(str).values
            hier_c = result['class'].astype(str)
            ca = (cast_c == hier_c).mean()
            print(f" | class agree: {100*ca:.1f}%", end='')

        print()

    # Save
    print(f"\nSaving to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    adata.write(output_path)

    # Summary
    print(f"\n{'='*60}")
    print(f"Annotation complete in {time.time()-t0:.0f}s")
    print(f"  Method: {method}")
    print(f"  Genes used: {n_genes}")
    print(f"  Cells: {adata.shape[0]:,}")
    print(f"  Samples: {len(samples)}")

    if 'class' in adata.obs.columns:
        ca = (adata.obs['class'].astype(str) == adata.obs[f'{prefix}_class'].astype(str)).mean()
        sa = (adata.obs['subclass'].astype(str) == adata.obs[f'{prefix}_subclass'].astype(str)).mean()
        print(f"  Overall CAST agreement: class={100*ca:.1f}%, subclass={100*sa:.1f}%")

    # Spatial coherence
    if 'x' in adata.obs.columns and 'y' in adata.obs.columns:
        coords = adata.obs[['x', 'y']].values
        sc_hier = _spatial_coherence(adata.obs[f'{prefix}_class'].values, coords).mean()
        print(f"  Spatial coherence (class): {sc_hier:.3f}")
        if 'class' in adata.obs.columns:
            sc_cast = _spatial_coherence(adata.obs['class'].values, coords).mean()
            print(f"  Spatial coherence CAST (class): {sc_cast:.3f}")

    print(f"{'='*60}")
    return adata


def main():
    parser = argparse.ArgumentParser(
        description='Annotate spatial transcriptomics data with Allen Brain Cell Atlas taxonomy')
    parser.add_argument('--input', required=True, help='Input h5ad file')
    parser.add_argument('--output', required=True, help='Output annotated h5ad file')
    parser.add_argument('--method', default='pearson', choices=['pearson', 'spearman'],
                        help='Correlation method (pearson for targeted panels, spearman for genome-wide)')
    parser.add_argument('--sample-col', default='sample', help='Column with sample IDs')
    parser.add_argument('--prefix', default='hier', help='Column prefix for annotations')
    parser.add_argument('--stats', default=os.path.join(WORKING_DIR, 'data/reference/precomputed_stats_ABC_revision_230821.h5'),
                        help='Path to precomputed stats HDF5')
    parser.add_argument('--ref', default=os.path.join(WORKING_DIR, 'output/data/adata_ref_zeng_imputed.h5ad'),
                        help='Path to Zeng reference h5ad (for valid type lists)')
    parser.add_argument('--merfish-mapping', default=os.path.join(WORKING_DIR, 'data/genename_mapping_merfish_panel.csv'),
                        help='Path to MERFISH gene panel mapping CSV')
    parser.add_argument('--gene-mapping', default=os.path.join(WORKING_DIR, 'data/reference/gene_symbol_to_ensembl_mouse.json'),
                        help='Path to gene symbol -> Ensembl ID JSON')
    args = parser.parse_args()

    annotate_dataset(
        input_path=args.input,
        output_path=args.output,
        stats_path=args.stats,
        valid_types_path=args.ref,
        merfish_mapping_path=args.merfish_mapping,
        symbol_to_ensembl_path=args.gene_mapping,
        method=args.method,
        sample_col=args.sample_col,
        prefix=args.prefix,
    )


if __name__ == '__main__':
    main()
