"""
Build snRNAseq-derived centroids from Allen Brain Cell Atlas precomputed stats.

This script extracts cluster-level mean expression from the precomputed stats HDF5,
aggregates to subclass and class levels, and saves centroid matrices that can be
used for correlation-based cell type classification.

Uses the MERFISH gene mapping (genename_mapping_merfish_panel.csv) to convert
between gene symbols and Ensembl IDs.

Output:
  - output/data/snrnaseq_centroids.npz: centroid matrices + metadata
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import h5py
from collections import defaultdict

# ── paths ────────────────────────────────────────────────────────────────
WORKING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(WORKING_DIR)

PRECOMPUTED_STATS = 'data/reference/precomputed_stats_ABC_revision_230821.h5'
MERFISH_GENE_MAPPING = 'data/genename_mapping_merfish_panel.csv'
GENE_SYMBOL_TO_ENSEMBL = 'data/reference/gene_symbol_to_ensembl_mouse.json'
OUTPUT_PATH = 'output/data/snrnaseq_centroids.npz'

os.makedirs('output/data', exist_ok=True)

def main():
    t0 = time.time()
    print("=" * 70)
    print("Building snRNAseq centroids from precomputed stats")
    print("=" * 70)

    # ── 1. Load precomputed stats ────────────────────────────────────────
    print("\n1. Loading precomputed stats...")
    f = h5py.File(PRECOMPUTED_STATS, 'r')

    ref_ensembl_ids = json.loads(f['col_names'][()])
    c2r = json.loads(f['cluster_to_row'][()])
    tree = json.loads(f['taxonomy_tree'][()])
    sums = f['sum'][:]       # (5322, 32285) — actually means since n_cells=1
    ncells = f['n_cells'][:]  # all 1s
    f.close()

    hierarchy = tree['hierarchy']
    nm = tree['name_mapper']
    print(f"   {sums.shape[0]} clusters x {sums.shape[1]} genes")
    print(f"   Hierarchy: {hierarchy}")

    # ── 2. Build taxonomy lookups ────────────────────────────────────────
    print("\n2. Building taxonomy lookups...")
    clas_level, subc_level, supt_level, clus_level = hierarchy

    clas_data = tree[clas_level]
    subc_data = tree[subc_level]
    supt_data = tree[supt_level]

    clus_to_subc = {}
    clus_to_class = {}
    clus_to_supt = {}
    subc_to_class = {}

    for clas_id, subc_ids in clas_data.items():
        clas_name = nm[clas_level][clas_id]['name']
        for subc_id in subc_ids:
            subc_name = nm[subc_level][subc_id]['name']
            subc_to_class[subc_name] = clas_name
            if subc_id in subc_data:
                for supt_id in subc_data[subc_id]:
                    supt_name = nm[supt_level][supt_id]['name']
                    if supt_id in supt_data:
                        for clus_id in supt_data[supt_id]:
                            clus_to_subc[clus_id] = subc_name
                            clus_to_class[clus_id] = clas_name
                            clus_to_supt[clus_id] = supt_name

    print(f"   {len(clus_to_subc)} clusters -> {len(set(clus_to_subc.values()))} subclasses "
          f"-> {len(set(clus_to_class.values()))} classes")

    # ── 3. Build gene mapping ────────────────────────────────────────────
    print("\n3. Building gene mapping...")

    # Primary: MERFISH panel mapping (symbol -> ensembl)
    merfish_map = pd.read_csv(MERFISH_GENE_MAPPING)
    merfish_sym2ens = dict(zip(
        merfish_map['Gene Symbol'],
        merfish_map['Gene ID '].str.strip()
    ))

    # Secondary: broader mapping from JSON
    with open(GENE_SYMBOL_TO_ENSEMBL) as gf:
        broad_sym2ens = json.load(gf)

    # Combine (MERFISH mapping takes priority)
    combined_sym2ens = {**broad_sym2ens, **merfish_sym2ens}

    # Build reverse: ensembl -> symbol
    ens2sym = {}
    for sym, ens in combined_sym2ens.items():
        ens2sym[ens] = sym

    # Map reference Ensembl IDs to symbols
    ref_symbols = [ens2sym.get(ens, ens) for ens in ref_ensembl_ids]
    n_mapped = sum(1 for s in ref_symbols if not s.startswith('ENSMUSG'))
    print(f"   {n_mapped}/{len(ref_symbols)} reference genes mapped to symbols")

    # ── 4. Build cluster-level centroid matrix ───────────────────────────
    print("\n4. Building cluster centroids...")
    # sums already contains means (n_cells=1 for all)
    cluster_means = sums  # (5322, 32285)

    cluster_ids = list(c2r.keys())
    cluster_names = [nm[clus_level].get(cid, {}).get('name', cid) for cid in cluster_ids]
    cluster_subclasses = [clus_to_subc.get(cid, 'Unknown') for cid in cluster_ids]
    cluster_classes = [clus_to_class.get(cid, 'Unknown') for cid in cluster_ids]

    # ── 5. Aggregate to subclass centroids ───────────────────────────────
    print("\n5. Aggregating to subclass centroids...")
    subc_sums = defaultdict(lambda: np.zeros(len(ref_ensembl_ids)))
    subc_counts = defaultdict(int)

    for i, cid in enumerate(cluster_ids):
        row = c2r[cid]
        sc = clus_to_subc.get(cid, 'Unknown')
        subc_sums[sc] += cluster_means[row]
        subc_counts[sc] += 1

    # Mean of cluster means (unweighted — each cluster contributes equally)
    subc_names = sorted([sc for sc in subc_sums if sc != 'Unknown'])
    subc_centroids = np.zeros((len(subc_names), len(ref_ensembl_ids)), dtype=np.float32)
    for i, sc in enumerate(subc_names):
        subc_centroids[i] = subc_sums[sc] / subc_counts[sc]

    print(f"   {len(subc_names)} subclass centroids")
    print(f"   Value range: [{subc_centroids.min():.3f}, {subc_centroids.max():.3f}]")

    # ── 6. Aggregate to class centroids ──────────────────────────────────
    print("\n6. Aggregating to class centroids...")
    clas_sums = defaultdict(lambda: np.zeros(len(ref_ensembl_ids)))
    clas_counts = defaultdict(int)

    for i, cid in enumerate(cluster_ids):
        row = c2r[cid]
        cc = clus_to_class.get(cid, 'Unknown')
        clas_sums[cc] += cluster_means[row]
        clas_counts[cc] += 1

    clas_names = sorted([cc for cc in clas_sums if cc != 'Unknown'])
    clas_centroids = np.zeros((len(clas_names), len(ref_ensembl_ids)), dtype=np.float32)
    for i, cc in enumerate(clas_names):
        clas_centroids[i] = clas_sums[cc] / clas_counts[cc]

    print(f"   {len(clas_names)} class centroids")

    # ── 7. Save everything ───────────────────────────────────────────────
    print("\n7. Saving centroids...")
    np.savez_compressed(
        OUTPUT_PATH,
        # Gene info
        ref_ensembl_ids=np.array(ref_ensembl_ids),
        ref_symbols=np.array(ref_symbols),
        # Subclass centroids
        subc_centroids=subc_centroids,
        subc_names=np.array(subc_names),
        # Class centroids
        clas_centroids=clas_centroids,
        clas_names=np.array(clas_names),
        # Taxonomy
        subc_to_class=json.dumps(subc_to_class),
        # MERFISH-specific mapping
        merfish_sym2ens=json.dumps(merfish_sym2ens),
    )

    fsize = os.path.getsize(OUTPUT_PATH) / 1e6
    print(f"   Saved to {OUTPUT_PATH} ({fsize:.1f} MB)")

    # ── 8. Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Classes: {len(clas_names)}")
    print(f"  Subclasses: {len(subc_names)}")
    print(f"  Genes: {len(ref_ensembl_ids)} ({n_mapped} with symbols)")
    print(f"  Time: {time.time()-t0:.1f}s")

    # Print class summary
    print(f"\n  Classes and their subclass counts:")
    for cc in clas_names:
        n_sub = sum(1 for sc in subc_names if subc_to_class.get(sc) == cc)
        print(f"    {cc}: {n_sub} subclasses")

if __name__ == '__main__':
    main()
