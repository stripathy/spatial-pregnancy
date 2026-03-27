"""
Step 1: Cell type annotation using Allen Institute's MapMyCells (HANN algorithm).

Adapted from SCZ_Xenium/code/pipeline/02_run_mapmycells.py for mouse brain data.

Uses bootstrapped HANN against the 10x Whole Mouse Brain taxonomy (CCN20230722)
to assign class (34), subclass (338), and supertype (1201) labels.

For each dataset (Slide-tags, MERFISH):
  1. Load h5ad with raw counts
  2. Convert gene symbols to Ensembl IDs
  3. Run MapMyCells HANN mapping (100 bootstrap iterations)
  4. Parse output: labels + confidence at class/subclass/supertype
  5. Add HANN labels to a copy of the h5ad, preserving Keon's CAST annotations

Requires:
  - Python 3.12 with cell_type_mapper >= 1.7.0 (~/venv312)
  - Precomputed stats: data/reference/precomputed_stats_ABC_revision_230821.h5
  - Gene mapping: data/reference/gene_symbol_to_ensembl_mouse.json
  - Marker genes: data/reference/mouse_markers_230821.json

Usage:
    ~/venv312/bin/python3 code/01_run_mapmycells.py --dataset slide_tags
    ~/venv312/bin/python3 code/01_run_mapmycells.py --dataset merfish
"""
import os
import sys
import time
import json
import argparse
import tempfile
import numpy as np
import pandas as pd
import anndata as ad
import h5py
import scipy.sparse as sparse

# ── paths ────────────────────────────────────────────────────────────────
WORKING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(WORKING_DIR)

PRECOMPUTED_STATS_PATH = os.path.join(
    WORKING_DIR, 'data/reference/precomputed_stats_ABC_revision_230821.h5')
GENE_MAPPING_PATH = os.path.join(
    WORKING_DIR, 'data/reference/gene_symbol_to_ensembl_mouse.json')
MARKER_PATH = os.path.join(
    WORKING_DIR, 'data/reference/mouse_markers_230821.json')

# MapMyCells settings (same as SCZ_Xenium)
MAPMYCELLS_BOOTSTRAP_ITER = 100
MAPMYCELLS_BOOTSTRAP_FACTOR = 0.5
MAPMYCELLS_N_PER_UTILITY = 30

# Dataset paths
DATASETS = {
    'slide_tags': {
        'input': 'data/GSE313279_slide_tags.h5ad',
        'output': 'output/data/slide_tags_hann.h5ad',
    },
    'merfish': {
        'input': 'data/merfish_clean/merfish_normalized/adata_query_merfish_pub.h5ad',
        'output': 'output/data/merfish_hann.h5ad',
    },
    'xenium5k': {
        'input': 'data/objf_raw_filt.h5ad',
        'output': 'output/data/xenium5k_hann.h5ad',
    },
}


# ── Gene mapping ────────────────────────────────────────────────────────

def load_gene_mapping():
    with open(GENE_MAPPING_PATH) as f:
        return json.load(f)


def convert_genes_to_ensembl(adata, gene_mapping):
    """Convert gene symbols to Ensembl IDs, dropping unmappable genes."""
    mappable = [g for g in adata.var_names if g in gene_mapping]
    adata_sub = adata[:, mappable].copy()
    adata_sub.var_names = [gene_mapping[g] for g in mappable]
    adata_sub.var_names_make_unique()
    return adata_sub


# ── Taxonomy helpers ────────────────────────────────────────────────────

def _load_taxonomy_tree():
    with h5py.File(PRECOMPUTED_STATS_PATH, 'r') as f:
        raw = f['taxonomy_tree'][()]
        tree = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
    return tree


def _build_taxonomy_lookups(tree):
    """Build supertype_id -> name/subclass/class lookups from taxonomy tree."""
    hierarchy = tree['hierarchy']
    nm = tree['name_mapper']

    clas_level = hierarchy[0]
    subc_level = hierarchy[1]
    supt_level = hierarchy[2]

    clas_data = tree[clas_level]
    subc_data = tree[subc_level]

    supt_to_name = {sid: info['name'] for sid, info in nm[supt_level].items()}
    supt_to_subclass = {}
    supt_to_class = {}

    for clas_id, subc_ids in clas_data.items():
        clas_name = nm[clas_level][clas_id]['name']
        for subc_id in subc_ids:
            subc_name = nm[subc_level][subc_id]['name']
            for supt_id in subc_data.get(subc_id, []):
                supt_to_subclass[supt_id] = subc_name
                supt_to_class[supt_id] = clas_name

    return supt_to_name, supt_to_subclass, supt_to_class


# ── MapMyCells runner ───────────────────────────────────────────────────

def run_mapmycells_on_adata(query_h5ad_path, output_dir, n_processors=1):
    """
    Run MapMyCells HANN mapping using pre-computed marker genes.

    Uses FromSpecifiedMarkersRunner since the mouse whole-brain
    precomputed_stats file doesn't contain 'sumsq'/'ge1' needed
    for on-the-fly marker selection.
    """
    from cell_type_mapper.cli.from_specified_markers import (
        FromSpecifiedMarkersRunner
    )

    hdf5_path = os.path.join(output_dir, "hann_output.h5")

    config = {
        "query_path": query_h5ad_path,
        "hdf5_result_path": hdf5_path,
        "precomputed_stats": {
            "path": PRECOMPUTED_STATS_PATH,
        },
        "query_markers": {
            "serialized_lookup": MARKER_PATH,
        },
        "type_assignment": {
            "normalization": "raw",
            "bootstrap_iteration": MAPMYCELLS_BOOTSTRAP_ITER,
            "bootstrap_factor": MAPMYCELLS_BOOTSTRAP_FACTOR,
            "algorithm": "hann",
            "n_processors": n_processors,
        },
        # Stop at subclass level — drop supertype and cluster levels
        # (class -> subclass only: 34 -> 338 types)
        "drop_level": "CCN20230722_SUPT",
        "flatten": False,
        "map_to_ensembl": False,
    }

    runner = FromSpecifiedMarkersRunner(args=[], input_data=config)
    runner.run()
    return hdf5_path


def parse_mapmycells_output(hdf5_path):
    """
    Parse HANN HDF5 output -> DataFrame with labels + confidence.

    With drop_level='CCN20230722_SUPT', the leaf level is subclass (338 types).
    Returns DataFrame with columns:
      hann_class, hann_subclass,
      hann_class_confidence, hann_subclass_confidence
    """
    tree = _load_taxonomy_tree()
    hierarchy = tree['hierarchy']
    nm = tree['name_mapper']

    clas_level = hierarchy[0]  # CCN20230722_CLAS
    subc_level = hierarchy[1]  # CCN20230722_SUBC

    # Build subclass -> class lookup
    clas_data = tree[clas_level]
    subc_to_name = {sid: info['name'] for sid, info in nm[subc_level].items()}
    subc_to_class = {}
    for clas_id, subc_ids in clas_data.items():
        clas_name = nm[clas_level][clas_id]['name']
        for subc_id in subc_ids:
            subc_to_class[subc_id] = clas_name

    with h5py.File(hdf5_path, 'r') as f:
        votes = f['votes'][:]
        correlation = f['correlation'][:]
        cluster_ids = [c.decode() if isinstance(c, bytes) else c
                       for c in f['cluster_identifiers'][:]]

    n_cells = votes.shape[0]
    print(f"  HANN output: {n_cells:,} cells x {votes.shape[1]} clusters "
          f"({len(cluster_ids)} cluster IDs)")

    # Pick cluster with most votes (correlation as tiebreaker)
    score = votes.astype(float) + correlation * 1e-6
    best_idx = np.argmax(score, axis=1)

    max_votes = votes[np.arange(n_cells), best_idx]
    total_votes = votes.sum(axis=1)
    subc_confidence = np.where(total_votes > 0,
                                max_votes / total_votes,
                                0.0).astype(np.float32)

    best_cluster_ids = [cluster_ids[i] for i in best_idx]

    result = pd.DataFrame(index=range(n_cells))

    # Subclass (leaf level with drop_level)
    result["hann_subclass"] = [subc_to_name.get(cid, cid)
                                for cid in best_cluster_ids]
    result["hann_subclass_confidence"] = subc_confidence

    # Class: derived from subclass, confidence = sum of votes for all
    # subclasses in same class
    class_labels = [subc_to_class.get(cid, "Unknown")
                    for cid in best_cluster_ids]
    result["hann_class"] = class_labels

    cluster_classes = [subc_to_class.get(cid, "") for cid in cluster_ids]
    unique_classes = sorted(set(cluster_classes))
    cls_to_idx = {c: i for i, c in enumerate(unique_classes)}
    n_classes = len(unique_classes)

    cls_votes = np.zeros((n_cells, n_classes), dtype=np.float32)
    for j, cid in enumerate(cluster_ids):
        cc = cluster_classes[j]
        if cc in cls_to_idx:
            cls_votes[:, cls_to_idx[cc]] += votes[:, j]

    cell_class_idx = [cls_to_idx.get(c, 0) for c in class_labels]
    cls_conf = np.array([
        cls_votes[i, cell_class_idx[i]] / max(total_votes[i], 1)
        for i in range(n_cells)
    ], dtype=np.float32)
    result["hann_class_confidence"] = cls_conf

    return result


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['slide_tags', 'merfish', 'xenium5k'],
                        required=True)
    parser.add_argument('--n-workers', type=int, default=1)
    args = parser.parse_args()

    ds = DATASETS[args.dataset]
    os.makedirs(os.path.dirname(ds['output']), exist_ok=True)

    print(f"{'='*70}")
    print(f"MapMyCells HANN: {args.dataset}")
    print(f"{'='*70}")

    # 1. Load data
    print(f"\nLoading {ds['input']}...")
    t0 = time.time()
    adata = ad.read_h5ad(ds['input'])
    print(f"  {adata.shape[0]:,} cells x {adata.shape[1]:,} genes "
          f"({time.time()-t0:.1f}s)")

    # 2. Convert genes to Ensembl IDs
    print("\nConverting gene symbols to Ensembl IDs...")
    gene_mapping = load_gene_mapping()
    adata_ensembl = convert_genes_to_ensembl(adata, gene_mapping)
    print(f"  Mapped: {adata_ensembl.shape[1]:,} / {adata.shape[1]:,} genes "
          f"({100*adata_ensembl.shape[1]/adata.shape[1]:.1f}%)")

    # 3. Save temporary h5ad with Ensembl IDs for MapMyCells
    with tempfile.TemporaryDirectory() as tmpdir:
        query_path = os.path.join(tmpdir, "query.h5ad")

        # MapMyCells needs raw counts in X
        adata_query = adata_ensembl.copy()
        if sparse.issparse(adata_query.X):
            adata_query.X = adata_query.X.astype(np.float32)
        adata_query.write(query_path)
        print(f"  Saved query to {query_path}")

        # 4. Run MapMyCells
        print(f"\nRunning MapMyCells HANN ({MAPMYCELLS_BOOTSTRAP_ITER} "
              f"bootstrap iterations)...")
        t0 = time.time()
        hdf5_path = run_mapmycells_on_adata(
            query_path, tmpdir, n_processors=args.n_workers)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed/60:.1f} minutes")

        # 5. Parse output
        print("\nParsing HANN output...")
        hann_df = parse_mapmycells_output(hdf5_path)

    # 6. Add HANN labels to original adata (preserving Keon's annotations)
    print("\nAdding HANN labels to adata...")
    for col in hann_df.columns:
        adata.obs[col] = hann_df[col].values

    # 7. Summary
    print(f"\n{'='*70}")
    print(f"HANN Annotation Summary ({args.dataset})")
    print(f"{'='*70}")
    print(f"\nClass confidence: "
          f"mean={adata.obs['hann_class_confidence'].mean():.3f}, "
          f"median={adata.obs['hann_class_confidence'].median():.3f}")
    print(f"Subclass confidence: "
          f"mean={adata.obs['hann_subclass_confidence'].mean():.3f}, "
          f"median={adata.obs['hann_subclass_confidence'].median():.3f}")

    print(f"\nClasses ({adata.obs['hann_class'].nunique()}):")
    for c, n in adata.obs['hann_class'].value_counts().head(10).items():
        print(f"  {c}: {n:,}")

    print(f"\nSubclasses ({adata.obs['hann_subclass'].nunique()}):")
    for c, n in adata.obs['hann_subclass'].value_counts().head(15).items():
        print(f"  {c}: {n:,}")

    # Compare with Keon's CAST annotations if present
    if 'class' in adata.obs.columns and 'subclass' in adata.obs.columns:
        print(f"\n--- Quick comparison with CAST annotations ---")
        class_agree = (adata.obs['hann_class'] == adata.obs['class']).mean()
        print(f"Class agreement: {100*class_agree:.1f}%")
        sub_agree = (adata.obs['hann_subclass'] == adata.obs['subclass']).mean()
        print(f"Subclass agreement: {100*sub_agree:.1f}%")

    # 8. Save
    print(f"\nSaving to {ds['output']}...")
    adata.write(ds['output'])
    print(f"Done!")


if __name__ == '__main__':
    main()
