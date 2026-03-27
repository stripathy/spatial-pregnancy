"""
Correlation-based cell type classifier for pregnancy spatial data.

Adapted from SCZ_Xenium pipeline. Builds per-subclass mean expression
centroids from query data (using Keon's CAST labels as ground truth),
then classifies all cells via Pearson correlation.

Can also use the Zeng ABCA imputed reference as centroids.

Usage:
    python code/02_correlation_classifier.py --dataset merfish --mode cast_centroids
    python code/02_correlation_classifier.py --dataset merfish --mode ref_centroids
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
from modules.correlation import correlate, assign_labels, build_centroids_from_labels
from modules.config import TOP_N_EXEMPLARS, RANDOM_SEED

DATASETS = {
    'merfish': {
        'input': 'data/merfish_clean/merfish_normalized/adata_query_merfish_pub.h5ad',
        'output': 'output/data/merfish_corr.h5ad',
    },
    'slide_tags': {
        'input': 'data/GSE313279_slide_tags.h5ad',
        'output': 'output/data/slide_tags_corr.h5ad',
    },
}

REF_IMPUTED_PATH = 'output/data/adata_ref_zeng_imputed.h5ad'


# ── Core functions ───────────────────────────────────────────────────────
# correlate(), assign_labels(), build_centroids_from_labels() imported from modules.correlation


def build_centroids_from_reference(ref_adata, subclass_col='subclass',
                                    top_n=TOP_N_EXEMPLARS):
    """Build centroids from the Zeng ABCA imputed reference.

    The reference X is already log2-transformed, so we just compute means.
    """
    labels = ref_adata.obs[subclass_col].astype(str).values
    unique_labels = sorted(set(labels))

    X = ref_adata.X
    if sparse.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    gene_names = list(ref_adata.var_names)

    centroids_dict = {}
    cell_counts = {}
    for lab in unique_labels:
        lab_mask = labels == lab
        n = lab_mask.sum()
        if n == 0:
            continue
        # Use up to top_n cells (random subset if too many)
        if n > top_n:
            rng = np.random.RandomState(42)
            idx = np.where(lab_mask)[0]
            idx = rng.choice(idx, top_n, replace=False)
            centroids_dict[lab] = X[idx].mean(axis=0)
            cell_counts[lab] = top_n
        else:
            centroids_dict[lab] = X[lab_mask].mean(axis=0)
            cell_counts[lab] = n

    centroids = pd.DataFrame(centroids_dict, index=gene_names).T
    print(f"  Built {len(centroids)} subclass centroids from reference "
          f"(top-{top_n} exemplars)")
    return centroids, cell_counts, gene_names


def run_classifier(adata, centroids, gene_names):
    """Run correlation classifier on all cells.

    Parameters
    ----------
    adata : AnnData
        Raw counts in .X
    centroids : pd.DataFrame
        (n_types, n_genes)
    gene_names : list
        Genes to use (intersection of query and centroid genes)

    Returns
    -------
    pd.DataFrame with corr_subclass, corr_subclass_corr, corr_subclass_margin
    """
    # Intersect genes
    query_genes = set(adata.var_names)
    centroid_genes = set(centroids.columns)
    shared_genes = sorted(query_genes & centroid_genes)
    print(f"  Shared genes: {len(shared_genes)} "
          f"(query: {len(query_genes)}, centroids: {len(centroid_genes)})")

    if len(shared_genes) < 50:
        raise ValueError(f"Too few shared genes ({len(shared_genes)})")

    # Subset to shared genes
    adata_sub = adata[:, shared_genes].copy()
    centroids_sub = centroids[shared_genes]

    # Normalize query
    print("  Normalizing query expression...")
    sc.pp.normalize_total(adata_sub, target_sum=1e4)
    sc.pp.log1p(adata_sub)

    X_query = adata_sub.X
    if sparse.issparse(X_query):
        X_query = X_query.toarray()
    X_query = X_query.astype(np.float32)
    X_query = np.nan_to_num(X_query, nan=0.0)

    # Correlate
    print(f"  Computing correlations ({adata.n_obs:,} cells x "
          f"{len(centroids)} types)...")
    t0 = time.time()
    corr_mat, type_names = correlate(X_query, centroids_sub)
    labels, best_corr, margin = assign_labels(corr_mat, type_names)
    print(f"  Done in {time.time()-t0:.1f}s")

    results = pd.DataFrame({
        'corr_subclass': labels,
        'corr_subclass_corr': best_corr,
        'corr_subclass_margin': margin,
    }, index=adata.obs.index)

    return results


def derive_class_from_subclass(subclass_labels, adata_ref=None,
                                class_col='class', subclass_col='subclass'):
    """Map subclass labels to class labels using a reference mapping."""
    if adata_ref is not None:
        # Build mapping from reference
        sub_to_class = dict(zip(
            adata_ref.obs[subclass_col].astype(str),
            adata_ref.obs[class_col].astype(str)
        ))
    else:
        sub_to_class = {}

    return np.array([sub_to_class.get(s, 'Unknown') for s in subclass_labels])


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['merfish', 'slide_tags'],
                        required=True)
    parser.add_argument('--mode', choices=['cast_centroids', 'ref_centroids'],
                        default='cast_centroids',
                        help='cast_centroids: build from CAST labels in query; '
                             'ref_centroids: build from Zeng reference')
    parser.add_argument('--sample', default=None,
                        help='Run on single sample (e.g., POSTPART_1)')
    args = parser.parse_args()

    ds = DATASETS[args.dataset]

    # 1. Load query data
    print(f"Loading {args.dataset} data...")
    adata = ad.read_h5ad(ds['input'])
    if args.sample:
        adata = adata[adata.obs['sample'] == args.sample].copy()
    print(f"  Shape: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")

    # 2. Build centroids
    if args.mode == 'cast_centroids':
        print(f"\nBuilding centroids from CAST labels...")
        centroids, counts, gene_names = build_centroids_from_labels(
            adata,
            subclass_col='subclass',
            confidence_col='subclass_confidence',
            top_n=200,
        )
    else:
        print(f"\nBuilding centroids from Zeng ABCA reference...")
        if not os.path.exists(REF_IMPUTED_PATH):
            print(f"  ERROR: Reference not found at {REF_IMPUTED_PATH}")
            print(f"  Run the reference prep script first.")
            sys.exit(1)
        ref = ad.read_h5ad(REF_IMPUTED_PATH)
        centroids, counts, gene_names = build_centroids_from_reference(
            ref, subclass_col='subclass', top_n=200)
        del ref

    # 3. Run classifier
    print(f"\nRunning correlation classifier ({args.mode})...")
    t0 = time.time()
    results = run_classifier(adata, centroids, gene_names)
    elapsed = time.time() - t0

    # 4. Derive class from subclass
    # Use the query data's own class/subclass mapping
    results['corr_class'] = derive_class_from_subclass(
        results['corr_subclass'].values, adata_ref=adata)

    # 5. Add to adata
    for col in results.columns:
        adata.obs[col] = results[col].values

    # 6. Save
    os.makedirs(os.path.dirname(ds['output']), exist_ok=True)
    out_path = ds['output']
    if args.sample:
        out_path = out_path.replace('.h5ad', f'_{args.sample}.h5ad')
    adata.write(out_path)
    print(f"\nSaved to {out_path}")

    # 7. Summary
    print(f"\n{'='*60}")
    print(f"Correlation Classifier Summary ({args.dataset}, {args.mode})")
    print(f"{'='*60}")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"\nCorrelation stats:")
    print(f"  Mean corr:   {results['corr_subclass_corr'].mean():.3f}")
    print(f"  Median corr: {results['corr_subclass_corr'].median():.3f}")
    print(f"  Mean margin: {results['corr_subclass_margin'].mean():.4f}")

    print(f"\nClasses ({results['corr_class'].nunique()}):")
    for c, n in results['corr_class'].value_counts().head(10).items():
        print(f"  {c}: {n:,}")

    print(f"\nSubclasses ({results['corr_subclass'].nunique()}):")
    for c, n in results['corr_subclass'].value_counts().head(15).items():
        print(f"  {c}: {n:,}")

    # Compare with CAST
    if 'subclass' in adata.obs.columns:
        print(f"\n--- Comparison with CAST annotations ---")
        cast_sub = adata.obs['subclass'].astype(str).values
        corr_sub = results['corr_subclass'].values
        agree = (cast_sub == corr_sub).mean()
        print(f"Subclass agreement (CAST vs Corr): {100*agree:.1f}%")

        if 'class' in adata.obs.columns:
            cast_cls = adata.obs['class'].astype(str).values
            corr_cls = results['corr_class'].values
            cls_agree = (cast_cls == corr_cls).mean()
            print(f"Class agreement (CAST vs Corr): {100*cls_agree:.1f}%")


if __name__ == '__main__':
    main()
