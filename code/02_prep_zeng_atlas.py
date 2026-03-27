"""
Prepare Zeng ABCA MERFISH reference atlas for CAST alignment.
Adapted from Keon Arbabi's merfish_prep_atlases.py

Selects target brain sections, merges cell metadata with expression data,
and saves standardized anndata objects (raw + imputed).
"""
import os
import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import warnings
import time
warnings.filterwarnings('ignore')

working_dir = '.'
os.makedirs(f'{working_dir}/output/data', exist_ok=True)
os.makedirs(f'{working_dir}/figures/reference', exist_ok=True)

# Load joined metadata from step 01
print("Loading joined metadata...")
cell_joined = pd.read_csv('ABC/metadata/cells_joined.csv')
print(f"  {cell_joined.shape[0]} cells, {cell_joined.shape[1]} columns")

# Target sections (same as Keon's analysis - hypothalamic region)
sections = [
    'C57BL6J-638850.49', 'C57BL6J-638850.48',
    'C57BL6J-638850.47', 'C57BL6J-638850.46'
]

# Process both imputed and raw expression data
# NOTE: These h5ad files are large and downloaded by abc_atlas_access
# The paths below match the ABC cache directory structure
data_types = [
    ('imputed',
     'ABC/expression_matrices/MERFISH-C57BL6J-638850-imputed/'
     '20240831/C57BL6J-638850-imputed-log2.h5ad',
     'adata_ref_zeng_imputed.h5ad'),
    ('raw',
     'ABC/expression_matrices/MERFISH-C57BL6J-638850/'
     '20230830/C57BL6J-638850-raw.h5ad',
     'adata_ref_zeng_raw.h5ad')
]

for data_type, input_path, output_filename in data_types:
    if not os.path.exists(input_path):
        print(f"\nSkipping {data_type} - file not found: {input_path}")
        print("  (Run download script first or check ABC cache paths)")
        continue

    print(f'\n--- Processing {data_type} data ---')
    t0 = time.time()
    adata_input = ad.read_h5ad(input_path)
    print(f"  Loaded: {adata_input.shape} ({time.time()-t0:.1f}s)")

    adatas_processed = []
    for section in sections:
        adata = adata_input[adata_input.obs['brain_section_label'] == section]
        adata.obs = adata.obs.reset_index()
        adata.obs = pd.merge(adata.obs, cell_joined, on='cell_label',
                             how='left')
        adata.obs = adata.obs.set_index('cell_label', drop=True)

        # Exclude unassigned cells
        exclude = ['unassigned', 'brain-unassigned',
                   'fiber tracts-unassigned']
        adata = adata[~adata.obs['parcellation_division'].isin(exclude)]
        adata = adata[adata.obs['x_ccf'].notna()]
        adata.var = adata.var.reset_index()

        # Flip CCF coords to match orientation (as in Keon's code)
        adata.obs['z_ccf'] *= -1
        adata.obs['y_ccf'] *= -1
        adata.obs['x'] = adata.obs['z_ccf']
        adata.obs['y'] = adata.obs['y_ccf']
        adata.obs['sample'] = section
        adata.obs['source'] = 'Zeng-ABCA-Reference'
        print(f'  [{section}] {adata.shape[0]} cells')
        adatas_processed.append(adata)

    adata_combined = ad.concat(adatas_processed, axis=0, merge='same')
    adata_combined.var = adata_input.var.reset_index().set_index('gene_symbol')
    adata_combined.var['gene_symbol'] = adata_combined.var.index
    adata_combined.var = adata_combined.var.rename_axis(None)
    adata_combined = adata_combined[
        :, ~adata_combined.var.index.duplicated(keep='first')]

    # Convert to sparse and save
    adata_combined = adata_combined.copy()
    adata_combined.X = sparse.csr_matrix(adata_combined.X.astype(np.float32))
    out_path = f'{working_dir}/output/data/{output_filename}'
    adata_combined.write(out_path)
    print(f'  Saved {data_type} -> {out_path} '
          f'({adata_combined.shape[0]} cells x {adata_combined.shape[1]} genes)')

    # Plot each section colored by cell type
    for selection in [
        'class_color', 'subclass_color', 'parcellation_division_color',
        'parcellation_structure_color']:
        if selection not in adata_combined.obs.columns:
            continue
        fig, axes = plt.subplots(1, 4, figsize=(25, 7))
        fig.suptitle(f'Zeng ABCA Reference ({data_type}) - {selection}',
                     fontsize=20)
        for ax, (sample, data) in zip(
                axes, adata_combined.obs.groupby('sample')):
            ax.scatter(data['x'], data['y'], s=0.8, c=data[selection])
            ax.set_title(sample, fontsize=16)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        fig_path = (f'{working_dir}/figures/reference/'
                    f'zeng_{data_type}_{selection}.png')
        plt.savefig(fig_path, dpi=200, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f'  Saved figure: {fig_path}')

print("\nDone! Reference atlases prepared.")
