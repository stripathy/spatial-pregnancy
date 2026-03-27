"""
Subset Zeng ABCA imputed expression to 4 target hypothalamic sections,
merge with joined metadata, and save as a single reference h5ad.

Adapted from Keon Arbabi's merfish_prep_atlases.py

Target sections (hypothalamic region):
  C57BL6J-638850.46, .47, .48, .49

Input:
  - C57BL6J-638850-imputed-log2.h5ad (full imputed expression)
  - ABC/metadata/cells_joined.csv (joined cell metadata)

Output:
  - output/data/adata_ref_zeng_imputed.h5ad (subset, annotated)
"""
import os
import sys
import time
import pandas as pd
import numpy as np
import anndata as ad
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── paths ────────────────────────────────────────────────────────────────────
working_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(working_dir)
print(f"Working directory: {working_dir}")

# Check for the imputed file in likely locations
imputed_candidates = [
    os.path.expanduser('~/Downloads/C57BL6J-638850-imputed-log2.h5ad'),
    'ABC/expression_matrices/MERFISH-C57BL6J-638850-imputed/20240831/'
    'C57BL6J-638850-imputed-log2.h5ad',
]
imputed_path = None
for p in imputed_candidates:
    if os.path.exists(p):
        imputed_path = p
        break

if imputed_path is None:
    print("ERROR: Could not find C57BL6J-638850-imputed-log2.h5ad")
    print("Checked:", imputed_candidates)
    sys.exit(1)

print(f"Using imputed file: {imputed_path}")
print(f"  Size: {os.path.getsize(imputed_path) / (1024**3):.2f} GB")

os.makedirs('output/data', exist_ok=True)
os.makedirs('figures/reference', exist_ok=True)

# ── target sections ──────────────────────────────────────────────────────────
sections = [
    'C57BL6J-638850.46', 'C57BL6J-638850.47',
    'C57BL6J-638850.48', 'C57BL6J-638850.49'
]

# ── load joined metadata ─────────────────────────────────────────────────────
print("\nLoading joined metadata...")
t0 = time.time()
cell_joined = pd.read_csv('ABC/metadata/cells_joined.csv')
print(f"  {cell_joined.shape[0]:,} cells, {cell_joined.shape[1]} columns "
      f"({time.time()-t0:.1f}s)")

# ── load imputed expression (this is the big one) ────────────────────────────
print(f"\nLoading imputed expression from {imputed_path}...")
print("  (This may take several minutes for a 47 GB file)")
t0 = time.time()
adata_input = ad.read_h5ad(imputed_path)
print(f"  Loaded: {adata_input.shape[0]:,} cells x {adata_input.shape[1]:,} "
      f"genes ({time.time()-t0:.1f}s)")
print(f"  dtype: {adata_input.X.dtype}, "
      f"sparse: {sparse.issparse(adata_input.X)}")

# ── subset to target sections ────────────────────────────────────────────────
print(f"\nSubsetting to {len(sections)} target sections...")
adatas_processed = []
for section in sections:
    t0 = time.time()
    mask = adata_input.obs['brain_section_label'] == section
    adata = adata_input[mask].copy()

    # merge metadata
    adata.obs = adata.obs.reset_index()
    adata.obs = pd.merge(adata.obs, cell_joined, on='cell_label', how='left')
    adata.obs = adata.obs.set_index('cell_label', drop=True)

    # exclude unassigned cells
    exclude = ['unassigned', 'brain-unassigned', 'fiber tracts-unassigned']
    adata = adata[~adata.obs['parcellation_division'].isin(exclude)]
    adata = adata[adata.obs['x_ccf'].notna()]
    adata.var = adata.var.reset_index()

    # flip CCF coords to match orientation (as in Keon's code)
    adata.obs['z_ccf'] = adata.obs['z_ccf'] * -1
    adata.obs['y_ccf'] = adata.obs['y_ccf'] * -1

    # spatial coordinates for downstream analysis
    adata.obs['x'] = adata.obs['z_ccf']
    adata.obs['y'] = adata.obs['y_ccf']
    adata.obs['sample'] = section
    adata.obs['source'] = 'Zeng-ABCA-Reference'

    print(f"  [{section}] {adata.shape[0]:,} cells ({time.time()-t0:.1f}s)")
    adatas_processed.append(adata)

# ── concatenate ──────────────────────────────────────────────────────────────
print("\nConcatenating sections...")
adata_combined = ad.concat(adatas_processed, axis=0, merge='same')
adata_combined.var = adata_input.var.reset_index().set_index('gene_symbol')
adata_combined.var['gene_symbol'] = adata_combined.var.index
adata_combined.var = adata_combined.var.rename_axis(None)
adata_combined = adata_combined[
    :, ~adata_combined.var.index.duplicated(keep='first')]

# convert to sparse float32 for efficiency
adata_combined = adata_combined.copy()
if not sparse.issparse(adata_combined.X):
    adata_combined.X = sparse.csr_matrix(adata_combined.X.astype(np.float32))
else:
    adata_combined.X = adata_combined.X.astype(np.float32)

print(f"\nFinal object: {adata_combined.shape[0]:,} cells x "
      f"{adata_combined.shape[1]:,} genes")

# ── save ─────────────────────────────────────────────────────────────────────
out_path = 'output/data/adata_ref_zeng_imputed.h5ad'
print(f"\nSaving to {out_path}...")
t0 = time.time()
adata_combined.write(out_path)
size_gb = os.path.getsize(out_path) / (1024**3)
print(f"  Saved ({size_gb:.2f} GB, {time.time()-t0:.1f}s)")

# ── summary stats ────────────────────────────────────────────────────────────
print("\n=== Summary ===")
print(f"  Cells: {adata_combined.shape[0]:,}")
print(f"  Genes: {adata_combined.shape[1]:,}")
print(f"  Sections: {adata_combined.obs['sample'].nunique()}")
for s in sections:
    n = (adata_combined.obs['sample'] == s).sum()
    print(f"    {s}: {n:,} cells")
print(f"  Classes: {adata_combined.obs['class'].nunique()}")
print(f"  Subclasses: {adata_combined.obs['subclass'].nunique()}")

# ── QC plots ─────────────────────────────────────────────────────────────────
print("\nGenerating reference plots...")
for selection in ['class_color', 'subclass_color',
                  'parcellation_division_color',
                  'parcellation_structure_color']:
    if selection not in adata_combined.obs.columns:
        print(f"  Skipping {selection} (not in obs)")
        continue
    fig, axes = plt.subplots(1, 4, figsize=(28, 7))
    label = selection.replace('_color', '').replace('_', ' ').title()
    fig.suptitle(f'Zeng ABCA Reference (imputed) — {label}', fontsize=22)
    for ax, (sample, data) in zip(
            axes, adata_combined.obs.groupby('sample')):
        ax.scatter(data['x'], data['y'], s=0.5, c=data[selection],
                   rasterized=True)
        ax.set_title(sample, fontsize=18)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
    plt.tight_layout()
    fig_path = f'figures/reference/zeng_imputed_{selection}.png'
    plt.savefig(fig_path, dpi=200, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"  Saved {fig_path}")

print("\nDone!")
