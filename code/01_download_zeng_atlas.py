"""
Download Zeng ABCA MERFISH reference atlas metadata and expression data.
Adapted from Keon Arbabi's merfish_zeng_download_atlas.py

Downloads from Allen Brain Cell Atlas S3 bucket (public, no login needed):
- Cell metadata with cluster annotations
- Reconstructed coordinates
- CCF coordinates
- Parcellation annotations
- Expression matrices (raw + imputed)
"""
import warnings
import os
import time
from pathlib import Path
from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache

warnings.filterwarnings("ignore")

# Local download directory
download_base = Path('ABC')
abc_cache = AbcProjectCache.from_cache_dir(download_base)

# Load the manifest (uses the 20240831 release like Keon's code)
print("Loading manifest...")
abc_cache.load_manifest('releases/20240831/manifest.json')
print("Manifest loaded successfully.")

# 1. Download cell metadata
print("\n--- Downloading cell metadata ---")
t0 = time.time()
cell = abc_cache.get_metadata_dataframe(
    directory='MERFISH-C57BL6J-638850',
    file_name='cell_metadata_with_cluster_annotation')
cell.rename(columns={'x': 'x_section', 'y': 'y_section', 'z': 'z_section'},
            inplace=True)
cell.set_index('cell_label', inplace=True)
print(f"  Cell metadata: {cell.shape[0]} cells, {cell.shape[1]} columns "
      f"({time.time()-t0:.1f}s)")
print(f"  Columns: {list(cell.columns[:10])}...")

# 2. Download reconstructed coordinates
print("\n--- Downloading reconstructed coordinates ---")
t0 = time.time()
reconstructed_coords = abc_cache.get_metadata_dataframe(
    directory='MERFISH-C57BL6J-638850-CCF',
    file_name='reconstructed_coordinates',
    dtype={"cell_label": str})
reconstructed_coords.rename(
    columns={'x': 'x_reconstructed', 'y': 'y_reconstructed',
             'z': 'z_reconstructed'},
    inplace=True)
reconstructed_coords.set_index('cell_label', inplace=True)
cell_joined = cell.join(reconstructed_coords, how='inner')
print(f"  Reconstructed coords: {reconstructed_coords.shape[0]} cells "
      f"({time.time()-t0:.1f}s)")

# 3. Download CCF coordinates
print("\n--- Downloading CCF coordinates ---")
t0 = time.time()
ccf_coords = abc_cache.get_metadata_dataframe(
    directory='MERFISH-C57BL6J-638850-CCF',
    file_name='ccf_coordinates',
    dtype={"cell_label": str})
ccf_coords.rename(columns={'x': 'x_ccf', 'y': 'y_ccf', 'z': 'z_ccf'},
                  inplace=True)
ccf_coords.drop(['parcellation_index'], axis=1, inplace=True)
ccf_coords.set_index('cell_label', inplace=True)
cell_joined = cell_joined.join(ccf_coords, how='inner')
print(f"  CCF coords: {ccf_coords.shape[0]} cells ({time.time()-t0:.1f}s)")

# 4. Download parcellation annotations
print("\n--- Downloading parcellation annotations ---")
t0 = time.time()
parcellation_annotation = abc_cache.get_metadata_dataframe(
    directory='Allen-CCF-2020',
    file_name='parcellation_to_parcellation_term_membership_acronym')
parcellation_annotation.set_index('parcellation_index', inplace=True)
parcellation_annotation.columns = [
    'parcellation_%s' % x for x in parcellation_annotation.columns]

parcellation_color = abc_cache.get_metadata_dataframe(
    directory='Allen-CCF-2020',
    file_name='parcellation_to_parcellation_term_membership_color')
parcellation_color.set_index('parcellation_index', inplace=True)
parcellation_color.columns = [
    'parcellation_%s' % x for x in parcellation_color.columns]

cell_joined = cell_joined.join(parcellation_annotation, on='parcellation_index')
cell_joined = cell_joined.join(parcellation_color, on='parcellation_index')
print(f"  Parcellation annotations joined ({time.time()-t0:.1f}s)")

# 5. Save joined metadata
os.makedirs('ABC/metadata', exist_ok=True)
cell_joined.to_csv('ABC/metadata/cells_joined.csv')
print(f"\n--- Saved joined metadata: {cell_joined.shape[0]} cells, "
      f"{cell_joined.shape[1]} columns ---")
print(f"  -> ABC/metadata/cells_joined.csv")

# 6. Download expression matrices
print("\n--- Getting expression data paths ---")
try:
    raw_path = abc_cache.get_data_path(
        'MERFISH-C57BL6J-638850', 'C57BL6J-638850/raw')
    print(f"  Raw expression path: {raw_path}")
except Exception as e:
    print(f"  Raw expression: {e}")

try:
    imputed_path = abc_cache.get_data_path(
        'MERFISH-C57BL6J-638850-imputed', 'C57BL6J-638850-imputed/log2')
    print(f"  Imputed expression path: {imputed_path}")
except Exception as e:
    print(f"  Imputed expression: {e}")

print("\n--- Summary of brain sections available ---")
if 'brain_section_label' in cell_joined.columns:
    sections = cell_joined['brain_section_label'].value_counts()
    print(f"  {len(sections)} unique sections")
    print(f"  Target sections for Keon's analysis:")
    for s in ['C57BL6J-638850.49', 'C57BL6J-638850.48',
              'C57BL6J-638850.47', 'C57BL6J-638850.46']:
        n = sections.get(s, 0)
        print(f"    {s}: {n} cells")

print("\n--- Cell type summary ---")
for col in ['class', 'subclass']:
    if col in cell_joined.columns:
        n_types = cell_joined[col].nunique()
        print(f"  {n_types} unique {col} types")

print("\nDone! Atlas metadata downloaded to ABC/")
