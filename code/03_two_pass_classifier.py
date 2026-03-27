"""
Two-pass cross-modal correlation classifier.

Pass 1: Build centroids from Slide-tags CAST labels (restricted to shared genes),
        classify MERFISH cells via Pearson correlation.
Pass 2: Select top-100 MERFISH exemplars per subclass from Pass 1,
        rebuild MERFISH-native centroids, reclassify all MERFISH cells.

Usage:
    python code/03_two_pass_classifier.py --sample POSTPART_1
    python code/03_two_pass_classifier.py  # all samples
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sparse

# ── paths ────────────────────────────────────────────────────────────────
WORKING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(WORKING_DIR)

# ── shared modules ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(WORKING_DIR, 'code'))
from modules.correlation import correlate, assign_labels, normalize_expr
from modules.config import TOP_N_EXEMPLARS, MIN_CELLS_PASS2

SLIDE_TAGS_PATH = 'data/GSE313279_slide_tags.h5ad'
MERFISH_PATH = 'data/merfish_clean/merfish_normalized/adata_query_merfish_pub.h5ad'
OUTPUT_DIR = 'output/data'


# ── Pass 1: Slide-tags centroids → MERFISH classification ───────────────

def build_slide_tags_centroids(adata_st, shared_genes, subclass_col='subclass',
                                confidence_col='subclass_confidence', top_n=200):
    """Build per-subclass centroids from Slide-tags, restricted to shared genes.

    Returns
    -------
    centroids : pd.DataFrame (n_subclasses, n_shared_genes)
    sub_to_class : dict mapping subclass name → class name
    """
    # Subset to shared genes
    adata_sub = adata_st[:, shared_genes].copy()

    labels = adata_sub.obs[subclass_col].astype(str).values
    confidences = adata_sub.obs[confidence_col].astype(float).values
    unique_labels = sorted(set(labels))

    # Select exemplar cells
    exemplar_indices = []
    cell_counts = {}
    for lab in unique_labels:
        lab_mask = np.where(labels == lab)[0]
        n_available = len(lab_mask)
        if n_available == 0:
            continue
        n_use = min(top_n, n_available)
        lab_conf = confidences[lab_mask]
        top_idx = lab_mask[np.argsort(lab_conf)[-n_use:]]
        exemplar_indices.append(top_idx)
        cell_counts[lab] = n_use

    all_idx = np.concatenate(exemplar_indices)
    adata_ex = adata_sub[all_idx].copy()

    # Normalize
    sc.pp.normalize_total(adata_ex, target_sum=1e4)
    sc.pp.log1p(adata_ex)

    X = adata_ex.X
    if sparse.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    ex_labels = adata_ex.obs[subclass_col].astype(str).values

    centroids_dict = {}
    for lab in sorted(cell_counts.keys()):
        lab_mask = ex_labels == lab
        centroids_dict[lab] = X[lab_mask].mean(axis=0)

    centroids = pd.DataFrame(centroids_dict, index=shared_genes).T

    # Build subclass → class mapping
    sub_to_class = dict(zip(
        adata_st.obs[subclass_col].astype(str),
        adata_st.obs['class'].astype(str)
    ))

    print(f"  Built {len(centroids)} Slide-tags centroids "
          f"({len(shared_genes)} genes, top-{top_n} exemplars)")
    for lab in sorted(cell_counts):
        n = cell_counts[lab]
        if n < 50:
            print(f"    WARNING: {lab}: only {n} cells")

    return centroids, sub_to_class


def run_pass1(adata_mf, centroids_st, shared_genes):
    """Classify MERFISH cells using Slide-tags centroids.

    Returns DataFrame with pass1_subclass, pass1_corr, pass1_margin.
    """
    print(f"\n  Pass 1: Classifying {adata_mf.n_obs:,} MERFISH cells "
          f"vs {len(centroids_st)} ST centroids...")

    # Subset MERFISH to shared genes and normalize
    adata_sub = adata_mf[:, shared_genes].copy()
    X_query = normalize_expr(adata_sub)
    X_query = np.nan_to_num(X_query, nan=0.0)

    # Correlate
    t0 = time.time()
    corr_mat, type_names = correlate(X_query, centroids_st)
    labels, best_corr, margin = assign_labels(corr_mat, type_names)
    print(f"    Done in {time.time()-t0:.1f}s")
    print(f"    Median corr: {np.median(best_corr):.3f}, "
          f"median margin: {np.median(margin):.4f}")

    return pd.DataFrame({
        'pass1_subclass': labels,
        'pass1_corr': best_corr,
        'pass1_margin': margin,
    }, index=adata_mf.obs.index)


# ── Pass 2: MERFISH self-refinement ─────────────────────────────────────

def build_merfish_centroids(adata_mf, pass1_results, shared_genes,
                             top_n=100, min_cells=20):
    """Build MERFISH-native centroids from Pass 1 best exemplars.

    For each Pass 1 subclass, select top_n cells with highest pass1_corr.

    Returns
    -------
    centroids : pd.DataFrame (n_subclasses, n_shared_genes)
    skipped : list of subclass names with too few cells
    """
    adata_sub = adata_mf[:, shared_genes].copy()
    labels = pass1_results['pass1_subclass'].values
    corrs = pass1_results['pass1_corr'].values
    unique_labels = sorted(set(labels))

    exemplar_indices = []
    cell_counts = {}
    skipped = []

    for lab in unique_labels:
        lab_mask = np.where(labels == lab)[0]
        n_available = len(lab_mask)

        if n_available < min_cells:
            skipped.append(lab)
            continue

        n_use = min(top_n, n_available)
        lab_corr = corrs[lab_mask]
        top_idx = lab_mask[np.argsort(lab_corr)[-n_use:]]
        exemplar_indices.append(top_idx)
        cell_counts[lab] = n_use

    if not exemplar_indices:
        raise ValueError("No subclasses had enough cells for Pass 2")

    all_idx = np.concatenate(exemplar_indices)
    adata_ex = adata_sub[all_idx].copy()

    # Normalize
    sc.pp.normalize_total(adata_ex, target_sum=1e4)
    sc.pp.log1p(adata_ex)

    X = adata_ex.X
    if sparse.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    # Need to get labels for exemplar cells
    ex_labels = labels[all_idx]

    centroids_dict = {}
    for lab in sorted(cell_counts.keys()):
        lab_mask = ex_labels == lab
        centroids_dict[lab] = X[lab_mask].mean(axis=0)

    centroids = pd.DataFrame(centroids_dict, index=shared_genes).T

    print(f"  Built {len(centroids)} MERFISH-native centroids "
          f"(top-{top_n} exemplars per subclass)")
    if skipped:
        print(f"    Skipped {len(skipped)} subclasses with <{min_cells} cells: "
              f"{skipped[:5]}{'...' if len(skipped) > 5 else ''}")

    return centroids, skipped


def run_pass2(adata_mf, centroids_mf, shared_genes, pass1_results, skipped):
    """Reclassify all MERFISH cells using MERFISH-native centroids.

    Cells whose Pass 1 subclass was skipped keep their Pass 1 label.
    """
    print(f"\n  Pass 2: Reclassifying {adata_mf.n_obs:,} MERFISH cells "
          f"vs {len(centroids_mf)} MF-native centroids...")

    adata_sub = adata_mf[:, shared_genes].copy()
    X_query = normalize_expr(adata_sub)
    X_query = np.nan_to_num(X_query, nan=0.0)

    t0 = time.time()
    corr_mat, type_names = correlate(X_query, centroids_mf)
    labels, best_corr, margin = assign_labels(corr_mat, type_names)
    print(f"    Done in {time.time()-t0:.1f}s")
    print(f"    Median corr: {np.median(best_corr):.3f}, "
          f"median margin: {np.median(margin):.4f}")

    # For cells whose Pass 1 subclass was skipped, fall back to Pass 1
    p1_labels = pass1_results['pass1_subclass'].values
    p1_corr = pass1_results['pass1_corr'].values
    p1_margin = pass1_results['pass1_margin'].values

    n_fallback = 0
    for i in range(len(labels)):
        if p1_labels[i] in skipped:
            labels[i] = p1_labels[i]
            best_corr[i] = p1_corr[i]
            margin[i] = p1_margin[i]
            n_fallback += 1

    if n_fallback > 0:
        print(f"    {n_fallback:,} cells kept Pass 1 label (skipped subclasses)")

    return pd.DataFrame({
        'pass2_subclass': labels,
        'pass2_corr': best_corr,
        'pass2_margin': margin,
    }, index=adata_mf.obs.index)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', default=None,
                        help='Single MERFISH sample (e.g., POSTPART_1)')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t_start = time.time()

    # ── Load data ────────────────────────────────────────────────────────
    print("Loading Slide-tags data...")
    adata_st = ad.read_h5ad(SLIDE_TAGS_PATH)
    print(f"  {adata_st.n_obs:,} cells x {adata_st.n_vars:,} genes")

    print("Loading MERFISH data...")
    adata_mf = ad.read_h5ad(MERFISH_PATH)
    if args.sample:
        adata_mf = adata_mf[adata_mf.obs['sample'] == args.sample].copy()
    print(f"  {adata_mf.n_obs:,} cells x {adata_mf.n_vars:,} genes")

    # ── Find shared genes ────────────────────────────────────────────────
    shared_genes = sorted(set(adata_st.var_names) & set(adata_mf.var_names))
    print(f"\nShared genes: {len(shared_genes)}")

    # ── Pass 1: Slide-tags → MERFISH ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("PASS 1: Slide-tags centroids → MERFISH classification")
    print(f"{'='*60}")

    centroids_st, sub_to_class = build_slide_tags_centroids(
        adata_st, shared_genes, top_n=200)

    pass1 = run_pass1(adata_mf, centroids_st, shared_genes)

    # ── Pass 2: MERFISH self-refinement ──────────────────────────────────
    print(f"\n{'='*60}")
    print("PASS 2: MERFISH self-refinement")
    print(f"{'='*60}")

    centroids_mf, skipped = build_merfish_centroids(
        adata_mf, pass1, shared_genes, top_n=100, min_cells=20)

    pass2 = run_pass2(adata_mf, centroids_mf, shared_genes, pass1, skipped)

    # ── Derive class labels ──────────────────────────────────────────────
    pass1['pass1_class'] = [sub_to_class.get(s, 'Unknown') for s in pass1['pass1_subclass']]
    pass2['pass2_class'] = [sub_to_class.get(s, 'Unknown') for s in pass2['pass2_subclass']]

    # ── Add to adata and save ────────────────────────────────────────────
    for col in pass1.columns:
        adata_mf.obs[col] = pass1[col].values
    for col in pass2.columns:
        adata_mf.obs[col] = pass2[col].values

    suffix = f'_{args.sample}' if args.sample else ''
    out_path = f'{OUTPUT_DIR}/merfish_two_pass{suffix}.h5ad'
    adata_mf.write(out_path)
    print(f"\nSaved to {out_path}")

    # ── Comparison ───────────────────────────────────────────────────────
    cast_cls = adata_mf.obs['class'].astype(str).values
    cast_sub = adata_mf.obs['subclass'].astype(str).values
    p1_cls = pass1['pass1_class'].values
    p1_sub = pass1['pass1_subclass'].values
    p2_cls = pass2['pass2_class'].values
    p2_sub = pass2['pass2_subclass'].values

    print(f"\n{'='*60}")
    print(f"RESULTS ({adata_mf.n_obs:,} cells)")
    print(f"{'='*60}")

    print(f"\n--- Overall agreement with CAST ---")
    print(f"  Pass 1 (ST→MF):  class={100*(cast_cls==p1_cls).mean():.1f}%  "
          f"subclass={100*(cast_sub==p1_sub).mean():.1f}%")
    print(f"  Pass 2 (MF→MF):  class={100*(cast_cls==p2_cls).mean():.1f}%  "
          f"subclass={100*(cast_sub==p2_sub).mean():.1f}%")
    print(f"  Pass1 vs Pass2:  class={100*(p1_cls==p2_cls).mean():.1f}%  "
          f"subclass={100*(p1_sub==p2_sub).mean():.1f}%")

    print(f"\n--- Correlation stats ---")
    print(f"  Pass 1: median corr={np.median(pass1['pass1_corr']):.3f}, "
          f"median margin={np.median(pass1['pass1_margin']):.4f}")
    print(f"  Pass 2: median corr={np.median(pass2['pass2_corr']):.3f}, "
          f"median margin={np.median(pass2['pass2_margin']):.4f}")

    print(f"\n--- Per-class agreement (CAST vs each pass) ---")
    df = pd.DataFrame({
        'cast_cls': cast_cls, 'p1_cls': p1_cls, 'p2_cls': p2_cls,
        'cast_sub': cast_sub, 'p1_sub': p1_sub, 'p2_sub': p2_sub,
    })
    for cls in sorted(df['cast_cls'].unique()):
        mask = df['cast_cls'] == cls
        n = mask.sum()
        p1_agree = (df.loc[mask, 'cast_cls'] == df.loc[mask, 'p1_cls']).mean()
        p2_agree = (df.loc[mask, 'cast_cls'] == df.loc[mask, 'p2_cls']).mean()
        p1_sub_agree = (df.loc[mask, 'cast_sub'] == df.loc[mask, 'p1_sub']).mean()
        p2_sub_agree = (df.loc[mask, 'cast_sub'] == df.loc[mask, 'p2_sub']).mean()
        print(f"  {cls:25s} ({n:6,}): "
              f"P1 cls={100*p1_agree:.0f}% sub={100*p1_sub_agree:.0f}%  |  "
              f"P2 cls={100*p2_agree:.0f}% sub={100*p2_sub_agree:.0f}%")

    print(f"\n--- Pass 2 subclass distribution (top 15) ---")
    for s, n in pd.Series(p2_sub).value_counts().head(15).items():
        print(f"  {s}: {n:,}")

    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")


if __name__ == '__main__':
    main()
