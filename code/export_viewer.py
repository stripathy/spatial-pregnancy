#!/usr/bin/env python3
"""
Export annotated spatial data for the interactive viewer.

Reads classified h5ad files from all 3 modalities and exports compact JSON
for the standalone HTML viewer. Then bundles everything into a single HTML file.

Usage:
    python code/export_viewer.py
"""

import os
import sys
import json
import gzip
import base64
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sparse

WORKING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(WORKING_DIR)

OUT_DIR = 'output/viewer'
os.makedirs(OUT_DIR, exist_ok=True)


def quantize_confidence(arr, scale=200):
    """Quantize float confidence to uint8 (0-200)."""
    return np.clip(np.round(np.array(arr, dtype=np.float32) * scale), 0, 255).astype(int).tolist()


def export_sample(obs, x_col, y_col, class_col, subclass_col, supertype_col,
                  class_conf_col, subclass_conf_col, sample_id, condition, modality,
                  flip_xy=False):
    """Export one sample to compact JSON dict."""
    x = obs[x_col].values.astype(np.float32)
    y = obs[y_col].values.astype(np.float32)

    # Flip 180 degrees (negate both x and y) for CCF-aligned data
    # so that cortex appears on top
    if flip_xy:
        x = -x
        y = -y

    # Build category lists
    class_cats = sorted(obs[class_col].unique())
    subclass_cats = sorted(obs[subclass_col].unique())
    supertype_cats = sorted(obs[supertype_col].unique()) if supertype_col in obs.columns else []

    class_idx = [class_cats.index(v) for v in obs[class_col]]
    subclass_idx = [subclass_cats.index(v) for v in obs[subclass_col]]
    supertype_idx = [supertype_cats.index(v) for v in obs[supertype_col]] if supertype_col in obs.columns else []

    # Choose rounding precision based on coordinate range
    # Small ranges (CCF coords ~10 units) need more decimals
    x_span = x.max() - x.min()
    decimals = 1 if x_span > 100 else (3 if x_span > 1 else 5)

    data = {
        'sample_id': sample_id,
        'condition': condition,
        'modality': modality,
        'n_cells': len(obs),
        'x_range': [float(x.min()), float(x.max())],
        'y_range': [float(y.min()), float(y.max())],
        'x': np.round(x, decimals).tolist(),
        'y': np.round(y, decimals).tolist(),
        'class_cats': class_cats,
        'subclass_cats': subclass_cats,
        'class': class_idx,
        'subclass': subclass_idx,
        'conf_class': quantize_confidence(obs[class_conf_col].values) if class_conf_col in obs.columns else [],
        'conf_subclass': quantize_confidence(obs[subclass_conf_col].values) if subclass_conf_col in obs.columns else [],
    }

    if supertype_cats:
        data['supertype_cats'] = supertype_cats
        data['supertype'] = supertype_idx

    data['grid'] = build_grid_index(x, y)

    return data


def load_colors():
    """Load Allen Institute cell type colors from Zeng reference."""
    ref = ad.read_h5ad('output/data/adata_ref_zeng_imputed.h5ad', backed='r')
    class_colors = ref.obs.groupby('class', observed=True)['class_color'].first().to_dict()
    subclass_colors = ref.obs.groupby('subclass', observed=True)['subclass_color'].first().to_dict()
    supertype_colors = ref.obs.groupby('supertype', observed=True)['supertype_color'].first().to_dict()
    ref.file.close()
    return class_colors, subclass_colors, supertype_colors


def main():
    print("Loading cell type colors...")
    class_colors, subclass_colors, supertype_colors = load_colors()

    samples_meta = []
    sample_jsons = {}

    # ── MERFISH ────────────────────────────────────────────────────────────
    print("\nExporting MERFISH samples...")
    merfish_dir = 'output/classification_v2'
    for f in sorted(os.listdir(merfish_dir)):
        if f.startswith('merfish_') and f.endswith('_classified.h5ad'):
            sample_name = f.replace('merfish_', '').replace('_classified.h5ad', '')
            print(f"  {sample_name}...", end='')

            adata = ad.read_h5ad(f'{merfish_dir}/{f}')
            condition = adata.obs['condition'].iloc[0]
            sid = f'merfish_{sample_name}'

            data = export_sample(
                adata.obs, 'x', 'y',
                'm3_class', 'm3_subclass', 'm3_supertype',
                'm3_class_conf', 'm3_subc_conf',
                sid, condition, 'MERFISH',
                flip_xy=True
            )

            # Also include CAST labels for comparison
            cast_class_cats = sorted(adata.obs['class'].unique())
            cast_subclass_cats = sorted(adata.obs['subclass'].unique())
            data['cast_class_cats'] = cast_class_cats
            data['cast_subclass_cats'] = cast_subclass_cats
            data['cast_class'] = [cast_class_cats.index(v) for v in adata.obs['class']]
            data['cast_subclass'] = [cast_subclass_cats.index(v) for v in adata.obs['subclass']]

            sample_jsons[sid] = data
            samples_meta.append({
                'sample_id': sid, 'condition': condition,
                'modality': 'MERFISH', 'n_cells': len(adata),
                'label': f'{sample_name} ({condition})'
            })
            print(f" {len(adata):,} cells")

    # ── Slide-tags ─────────────────────────────────────────────────────────
    print("\nExporting Slide-tags samples...")
    for f in sorted(os.listdir(merfish_dir)):
        if f.startswith('slidetags_') and f.endswith('_spearman.h5ad'):
            sample_name = f.replace('slidetags_', '').replace('_spearman.h5ad', '')
            print(f"  {sample_name}...", end='')

            adata = ad.read_h5ad(f'{merfish_dir}/{f}')
            condition = sample_name.split('_')[0]
            sid = f'slidetags_{sample_name}'

            data = export_sample(
                adata.obs, 'x', 'y',
                'm3_class', 'm3_subclass', 'm3_supertype',
                'm3_class_conf', 'm3_subc_conf',
                sid, condition, 'Slide-tags',
                flip_xy=True
            )

            # CAST labels
            cast_class_cats = sorted(adata.obs['class'].unique())
            cast_subclass_cats = sorted(adata.obs['subclass'].unique())
            data['cast_class_cats'] = cast_class_cats
            data['cast_subclass_cats'] = cast_subclass_cats
            data['cast_class'] = [cast_class_cats.index(v) for v in adata.obs['class']]
            data['cast_subclass'] = [cast_subclass_cats.index(v) for v in adata.obs['subclass']]

            sample_jsons[sid] = data
            samples_meta.append({
                'sample_id': sid, 'condition': condition,
                'modality': 'Slide-tags', 'n_cells': len(adata),
                'label': f'{sample_name} ({condition})'
            })
            print(f" {len(adata):,} cells")

    # ── Xenium 5k ──────────────────────────────────────────────────────────
    print("\nExporting Xenium 5k samples...")
    xenium_path = 'output/xenium5k/xenium5k_annotated.h5ad'
    if os.path.exists(xenium_path):
        adata = ad.read_h5ad(xenium_path)
        for sample_name in sorted(adata.obs['sample'].unique()):
            print(f"  {sample_name}...", end='')
            query = adata[adata.obs['sample'] == sample_name]
            condition = query.obs['condition'].iloc[0]
            sid = f'xenium5k_{sample_name}'

            # Xenium uses x_centroid/y_centroid
            data = export_sample(
                query.obs, 'x_centroid', 'y_centroid',
                'hier_class', 'hier_subclass', 'hier_subclass',  # no supertype yet
                'hier_class_conf', 'hier_subclass_conf',
                sid, condition, 'Xenium 5k'
            )

            sample_jsons[sid] = data
            samples_meta.append({
                'sample_id': sid, 'condition': condition,
                'modality': 'Xenium 5k', 'n_cells': len(query),
                'label': f'{sample_name} ({condition})'
            })
            print(f" {len(query):,} cells")

    # ── Zeng reference section ─────────────────────────────────────────────
    print("\nExporting Zeng reference section (.46)...")
    ref = ad.read_h5ad('output/data/adata_ref_zeng_imputed.h5ad')
    ref.var_names_make_unique()
    ref46 = ref[ref.obs['sample'] == 'C57BL6J-638850.46']
    sid = 'ref_zeng_46'

    ref_class_cats = sorted(ref46.obs['class'].unique())
    ref_subclass_cats = sorted(ref46.obs['subclass'].unique())
    ref_supertype_cats = sorted(ref46.obs['supertype'].unique())

    # Flip reference coords too (same CCF space)
    ref_x = -ref46.obs['x'].values.astype(np.float32)
    ref_y = -ref46.obs['y'].values.astype(np.float32)
    ref_data = {
        'sample_id': sid,
        'condition': 'Reference',
        'modality': 'Zeng MERFISH',
        'n_cells': len(ref46),
        'x_range': [float(ref_x.min()), float(ref_x.max())],
        'y_range': [float(ref_y.min()), float(ref_y.max())],
        'x': np.round(ref_x, 3).tolist(),
        'y': np.round(ref_y, 3).tolist(),
        'class_cats': ref_class_cats,
        'subclass_cats': ref_subclass_cats,
        'supertype_cats': ref_supertype_cats,
        'class': [ref_class_cats.index(v) for v in ref46.obs['class']],
        'subclass': [ref_subclass_cats.index(v) for v in ref46.obs['subclass']],
        'supertype': [ref_supertype_cats.index(v) for v in ref46.obs['supertype']],
        'conf_class': [],
        'conf_subclass': [],
    }
    sample_jsons[sid] = ref_data
    samples_meta.insert(0, {
        'sample_id': sid, 'condition': 'Reference',
        'modality': 'Zeng MERFISH', 'n_cells': len(ref46),
        'label': 'Zeng Reference (.46)'
    })
    print(f"  {len(ref46):,} cells")

    # ── Build index ────────────────────────────────────────────────────────
    index_data = {
        'samples': samples_meta,
        'class_colors': class_colors,
        'subclass_colors': subclass_colors,
        'supertype_colors': supertype_colors,
    }

    # Save raw JSONs
    with open(f'{OUT_DIR}/index.json', 'w') as f:
        json.dump(index_data, f)

    for sid, data in sample_jsons.items():
        with open(f'{OUT_DIR}/{sid}.json', 'w') as f:
            json.dump(data, f)

    print(f"\nExported {len(sample_jsons)} samples to {OUT_DIR}/")
    print(f"Total cells: {sum(s['n_cells'] for s in samples_meta):,}")

    # ── Bundle into standalone HTML ────────────────────────────────────────
    print("\nBundling into standalone HTML...")
    bundle_standalone(OUT_DIR, sample_jsons, index_data)

    print("Done!")


def build_grid_index(x, y, grid_size=100):
    """Build a spatial grid index for O(1) tooltip hit-testing in the viewer."""
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    x_span = x_max - x_min + 1e-9
    y_span = y_max - y_min + 1e-9

    bx = np.clip(((x - x_min) / x_span * grid_size).astype(int), 0, grid_size - 1)
    by = np.clip(((y - y_min) / y_span * grid_size).astype(int), 0, grid_size - 1)

    buckets = [[] for _ in range(grid_size * grid_size)]
    for i in range(len(x)):
        buckets[by[i] * grid_size + bx[i]].append(i)

    return {
        'grid_size': grid_size,
        'x_min': x_min, 'x_max': x_max,
        'y_min': y_min, 'y_max': y_max,
        'buckets': buckets,
    }


def compress_b64(data_dict):
    """Compress JSON dict to gzipped base64 string."""
    json_bytes = json.dumps(data_dict, separators=(',', ':')).encode('utf-8')
    compressed = gzip.compress(json_bytes, compresslevel=9)
    return base64.b64encode(compressed).decode('ascii')


def bundle_standalone(out_dir, sample_jsons, index_data):
    """Bundle all data into a single standalone HTML file."""

    # Compress all data
    index_b64 = compress_b64(index_data)
    sample_blobs = {}
    for sid, data in sample_jsons.items():
        sample_blobs[sid] = compress_b64(data)
        print(f"  {sid}: {len(sample_blobs[sid])//1024:,} KB compressed")

    # Read the viewer template
    viewer_html = generate_viewer_html(index_b64, sample_blobs)

    output_path = f'{out_dir}/pregnancy_viewer_standalone.html'
    with open(output_path, 'w') as f:
        f.write(viewer_html)

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"\nStandalone viewer: {output_path} ({size_mb:.1f} MB)")


def generate_viewer_html(index_b64, sample_blobs):
    """Generate the full standalone HTML viewer from viewer_template.html."""
    template_path = Path(__file__).parent / 'viewer_template.html'
    template = template_path.read_text(encoding='utf-8')

    blobs_js = ',\n'.join(f'    "{sid}": "{b64}"' for sid, b64 in sample_blobs.items())

    return (template
            .replace('__INDEX_B64__', index_b64)
            .replace('__SAMPLE_BLOBS__', blobs_js))


if __name__ == '__main__':
    main()
