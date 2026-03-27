"""
Memory-efficient extraction of Zeng ABCA imputed reference.

Reads the 47 GB HDF5 file directly via h5py, extracting only:
- 4 target brain sections (~380K cells)
- Union of MERFISH + Xenium 5k gene panels (~2,643 genes)

Peak memory: ~4-6 GB (safe on 24 GB machine).

Usage:
    python code/04_extract_zeng_reference.py
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import anndata as ad
import h5py
import scipy.sparse as sparse

# ── paths ────────────────────────────────────────────────────────────────
WORKING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(WORKING_DIR)

IMPUTED_PATH = ('ABC/expression_matrices/MERFISH-C57BL6J-638850/'
                '20230830/C57BL6J-638850-imputed-log2.h5ad')
METADATA_PATH = 'ABC/metadata/cells_joined.csv'
MERFISH_PATH = 'data/merfish_clean/merfish_normalized/adata_query_merfish_pub.h5ad'
XENIUM5K_PATH = 'data/objf_raw_filt.h5ad'
OUTPUT_PATH = 'output/data/adata_ref_zeng_imputed.h5ad'

TARGET_SECTIONS = [
    'C57BL6J-638850.46', 'C57BL6J-638850.47',
    'C57BL6J-638850.48', 'C57BL6J-638850.49',
]


def get_target_genes():
    """Get union of MERFISH + Xenium 5k gene panels."""
    genes = set()

    if os.path.exists(MERFISH_PATH):
        mf = ad.read_h5ad(MERFISH_PATH, backed='r')
        genes |= set(mf.var_names)
        print(f"  MERFISH panel: {len(mf.var_names)} genes")
        mf.file.close()

    if os.path.exists(XENIUM5K_PATH):
        x5k = ad.read_h5ad(XENIUM5K_PATH, backed='r')
        genes |= set(x5k.var_names)
        print(f"  Xenium 5k panel: {len(x5k.var_names)} genes")
        x5k.file.close()

    print(f"  Union: {len(genes)} unique genes")
    return genes


def phase1_discover_indices(h5file, target_sections, target_genes):
    """Find row indices (target sections) and column indices (target genes).

    Returns
    -------
    target_rows : np.ndarray of int
        Sorted row indices for target cells
    cell_labels : list of str
        Cell labels in same order as target_rows
    section_labels : list of str
        Section labels in same order as target_rows
    gene_col_idx : np.ndarray of int
        Column indices for target genes
    gene_names : list of str
        Gene names in same order as gene_col_idx
    """
    print("\nPhase 1: Discovering indices...")
    t0 = time.time()

    # Row indices
    target_set = set(target_sections)
    bsl = h5file['obs']['brain_section_label']
    total_rows = bsl.shape[0]

    target_rows = []
    section_labels = []
    chunk_size = 500000

    for start in range(0, total_rows, chunk_size):
        end = min(start + chunk_size, total_rows)
        chunk = bsl[start:end]
        chunk_str = [x.decode() if isinstance(x, bytes) else x for x in chunk]
        for i, s in enumerate(chunk_str):
            if s in target_set:
                target_rows.append(start + i)
                section_labels.append(s)
        elapsed = time.time() - t0
        print(f"  Scanned {end:,}/{total_rows:,} rows "
              f"({len(target_rows):,} target cells, {elapsed:.0f}s)", flush=True)

    target_rows = np.array(target_rows, dtype=np.int64)

    # Cell labels
    cell_label_ds = h5file['obs']['cell_label']
    print(f"  Reading {len(target_rows):,} cell labels...")
    # Read in batches to avoid huge fancy index
    cell_labels = []
    for batch_start in range(0, len(target_rows), 100000):
        batch_end = min(batch_start + 100000, len(target_rows))
        batch_idx = target_rows[batch_start:batch_end]
        batch_labels = cell_label_ds[batch_idx]
        cell_labels.extend(
            [x.decode() if isinstance(x, bytes) else x for x in batch_labels])

    # Gene column indices
    all_gene_syms = [x.decode() if isinstance(x, bytes) else x
                     for x in h5file['var']['gene_symbol'][:]]

    gene_col_idx = []
    gene_names = []
    for i, g in enumerate(all_gene_syms):
        if g in target_genes:
            gene_col_idx.append(i)
            gene_names.append(g)
    gene_col_idx = np.array(gene_col_idx, dtype=np.int64)

    elapsed = time.time() - t0
    print(f"  Found {len(target_rows):,} target cells, "
          f"{len(gene_col_idx)} target genes in {elapsed:.0f}s")

    for s in target_sections:
        n = section_labels.count(s)
        print(f"    {s}: {n:,}")

    return target_rows, cell_labels, section_labels, gene_col_idx, gene_names


def phase2_extract_expression(h5file, target_rows, gene_col_idx, batch_size=5000):
    """Extract expression matrix for target rows and genes.

    Reads full rows in sorted batches, then subsets columns in memory.

    Returns
    -------
    X : np.ndarray (n_target_cells, n_target_genes), float16
    """
    print(f"\nPhase 2: Extracting expression ({len(target_rows):,} cells × "
          f"{len(gene_col_idx)} genes)...")
    t0 = time.time()

    X_ds = h5file['X']
    n_cells = len(target_rows)
    n_genes = len(gene_col_idx)

    # Pre-allocate output
    out = np.zeros((n_cells, n_genes), dtype=np.float16)

    # Sort rows for sequential access
    sort_order = np.argsort(target_rows)
    sorted_rows = target_rows[sort_order]

    for batch_start in range(0, n_cells, batch_size):
        batch_end = min(batch_start + batch_size, n_cells)
        batch_sorted_idx = sorted_rows[batch_start:batch_end]

        # Read full rows for this batch
        # h5py fancy indexing with sorted array
        chunk = X_ds[batch_sorted_idx, :]  # (batch, 8460), float16

        # Subset to target genes (in-memory, fast)
        out[batch_start:batch_end] = chunk[:, gene_col_idx]

        elapsed = time.time() - t0
        rate = batch_end / elapsed if elapsed > 0 else 0
        eta = (n_cells - batch_end) / rate if rate > 0 else 0
        if batch_end % 50000 < batch_size or batch_end == n_cells:
            print(f"  Extracted {batch_end:,}/{n_cells:,} cells "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)", flush=True)

    # Unsort to match original target_rows order
    unsort_order = np.argsort(sort_order)
    out = out[unsort_order]

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  X shape: {out.shape}, dtype: {out.dtype}")
    print(f"  Memory: {out.nbytes / 1e9:.2f} GB")

    return out


def phase3_build_anndata(X, cell_labels, section_labels, gene_names,
                          target_sections):
    """Build AnnData with metadata from cells_joined.csv.

    Follows Keon's merfish_prep_atlases.py conventions:
    - Filter unassigned parcellations
    - Set CCF spatial coordinates (z_ccf → x, y_ccf → y, both negated)
    """
    print(f"\nPhase 3: Building AnnData...")
    t0 = time.time()

    # Load metadata
    print(f"  Loading cells_joined.csv...")
    cell_joined = pd.read_csv(METADATA_PATH)
    print(f"    {cell_joined.shape[0]:,} rows, {cell_joined.shape[1]} columns")

    # Build obs DataFrame
    obs = pd.DataFrame({
        'cell_label': cell_labels,
        'brain_section_label': section_labels,
    })

    # Merge with metadata
    obs = pd.merge(obs, cell_joined, on='cell_label', how='left',
                   suffixes=('', '_meta'))

    # Use brain_section_label from our data (not metadata)
    if 'brain_section_label_meta' in obs.columns:
        obs.drop('brain_section_label_meta', axis=1, inplace=True)

    # Filter unassigned parcellations
    exclude = ['unassigned', 'brain-unassigned', 'fiber tracts-unassigned']
    before = len(obs)
    keep_mask = ~obs['parcellation_division'].isin(exclude)
    keep_mask &= obs['x_ccf'].notna()
    obs_filt = obs[keep_mask].copy()
    X_filt = X[keep_mask.values]
    after = len(obs_filt)
    print(f"  Filtered: {before:,} → {after:,} cells "
          f"({before - after:,} removed)")

    # Set spatial coordinates (Keon's convention)
    obs_filt['z_ccf'] = obs_filt['z_ccf'] * -1
    obs_filt['y_ccf'] = obs_filt['y_ccf'] * -1
    obs_filt['x'] = obs_filt['z_ccf']
    obs_filt['y'] = obs_filt['y_ccf']
    obs_filt['source'] = 'Zeng-ABCA-Reference'
    obs_filt['sample'] = obs_filt['brain_section_label']

    # Set index
    obs_filt = obs_filt.set_index('cell_label', drop=True)
    obs_filt.index.name = None

    # Build var
    var = pd.DataFrame({'gene_symbol': gene_names}, index=gene_names)

    # Convert X to sparse float32
    print(f"  Converting to sparse float32...")
    X_sparse = sparse.csr_matrix(X_filt.astype(np.float32))

    # Build AnnData
    adata = ad.AnnData(X=X_sparse, obs=obs_filt, var=var)

    elapsed = time.time() - t0
    print(f"  AnnData: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes "
          f"(built in {elapsed:.0f}s)")

    # Summary
    for s in target_sections:
        n = (adata.obs['sample'] == s).sum()
        print(f"    {s}: {n:,}")

    print(f"\n  Class distribution:")
    if 'class' in adata.obs.columns:
        for c, n in adata.obs['class'].value_counts().head(10).items():
            print(f"    {c}: {n:,}")

    print(f"\n  Subclass count: {adata.obs['subclass'].nunique()}")

    return adata


def main():
    t_start = time.time()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Get target genes
    print("Getting target gene panels...")
    target_genes = get_target_genes()

    # Open HDF5 file
    print(f"\nOpening {IMPUTED_PATH}...")
    with h5py.File(IMPUTED_PATH, 'r') as f:
        print(f"  X shape: {f['X'].shape}, dtype: {f['X'].dtype}")

        # Phase 1: Find indices
        target_rows, cell_labels, section_labels, gene_col_idx, gene_names = \
            phase1_discover_indices(f, TARGET_SECTIONS, target_genes)

        # Phase 2: Extract expression
        X = phase2_extract_expression(f, target_rows, gene_col_idx,
                                       batch_size=5000)

    # Phase 3: Build AnnData (file closed, frees h5py cache)
    adata = phase3_build_anndata(X, cell_labels, section_labels, gene_names,
                                  TARGET_SECTIONS)

    # Free the raw X buffer
    del X

    # Phase 4: Save
    print(f"\nPhase 4: Saving to {OUTPUT_PATH}...")
    adata.write(OUTPUT_PATH)
    print(f"  Saved! File size: "
          f"{os.path.getsize(OUTPUT_PATH) / 1e6:.0f} MB")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed/60:.1f} minutes")


if __name__ == '__main__':
    main()
