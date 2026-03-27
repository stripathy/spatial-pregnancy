#!/usr/bin/env python3
"""
Build crumblr input CSVs for pregnancy compositional analysis.

Generates long-format count tables from classified MERFISH and Slide-tags data.
Each CSV: donor, celltype, count, total, condition

Strata: whole, neuronal-only, non-neuronal-only
Annotation methods: CAST, hierarchical correlation (m3)
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import time

WORKING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(WORKING_DIR, 'output', 'crumblr')
CLASS_DIR = os.path.join(WORKING_DIR, 'output', 'classification_v2')
os.makedirs(OUT_DIR, exist_ok=True)

# Neuronal class prefixes (from Allen taxonomy)
NEURONAL_KEYWORDS = ['Glut', 'GABA', 'Dopa', 'Sero', 'Gnrh1']

def is_neuronal(class_label):
    return any(kw in str(class_label) for kw in NEURONAL_KEYWORDS)


def build_counts(obs_df, celltype_col, class_col=None, stratum=None,
                  condition_col='condition', animal_col=None):
    """Build long-format count table."""
    if stratum == 'neuronal' and class_col:
        obs_df = obs_df[obs_df[class_col].apply(is_neuronal)]
    elif stratum == 'nonneuronal' and class_col:
        obs_df = obs_df[~obs_df[class_col].apply(is_neuronal)]

    if len(obs_df) == 0:
        return pd.DataFrame()

    counts = obs_df.groupby(['sample', celltype_col]).size().reset_index(name='count')
    totals = obs_df.groupby('sample').size().reset_index(name='total')
    counts = counts.merge(totals, on='sample')
    counts = counts.rename(columns={'sample': 'donor', celltype_col: 'celltype'})

    # Add condition
    if condition_col in obs_df.columns:
        cond_map = obs_df.groupby('sample')[condition_col].first().to_dict()
        counts['condition'] = counts['donor'].map(cond_map)
    else:
        counts['condition'] = counts['donor'].str.extract(r'^(CTRL|PREG|POSTPART)')[0]

    # Add animal (biological replicate) if available
    if animal_col and animal_col in obs_df.columns:
        animal_map = obs_df.groupby('sample')[animal_col].first().to_dict()
        counts['animal'] = counts['donor'].map(animal_map)

    return counts


def process_dataset(name, h5ad_paths, annotation_methods, min_presence=0.5,
                    condition_col='condition', animal_col=None):
    """Process one dataset with multiple annotation methods."""

    print(f"\n{'='*60}")
    print(f"Processing {name}")
    print(f"{'='*60}")

    # Load all samples
    all_obs = []
    for path in h5ad_paths:
        adata = ad.read_h5ad(path, backed='r')
        obs = adata.obs.copy()
        adata.file.close()
        all_obs.append(obs)

    obs_all = pd.concat(all_obs, ignore_index=False)
    n_samples = obs_all['sample'].nunique()
    print(f"  Total: {len(obs_all):,} cells, {n_samples} samples")
    if animal_col and animal_col in obs_all.columns:
        n_animals = obs_all[animal_col].nunique()
        print(f"  Animals (biological replicates): {n_animals}")

    for method_name, celltype_col, class_col in annotation_methods:
        print(f"\n  --- Annotation: {method_name} ---")

        for stratum in [None, 'neuronal', 'nonneuronal']:
            stratum_suffix = f'_{stratum}' if stratum else ''
            stratum_label = stratum or 'whole'

            counts = build_counts(obs_all, celltype_col, class_col, stratum,
                                  condition_col=condition_col, animal_col=animal_col)
            if len(counts) == 0:
                print(f"    {stratum_label}: no cells, skipping")
                continue

            # Filter: cell types present in >= min_presence of samples
            presence = counts.groupby('celltype')['donor'].nunique() / n_samples
            keep_types = presence[presence >= min_presence].index
            counts = counts[counts['celltype'].isin(keep_types)]

            n_types = counts['celltype'].nunique()
            outpath = os.path.join(OUT_DIR,
                f'crumblr_input_{name}_{method_name}{stratum_suffix}.csv')
            counts.to_csv(outpath, index=False)
            print(f"    {stratum_label}: {n_samples} donors x {n_types} types "
                  f"-> {os.path.basename(outpath)}")


def main():
    t0 = time.time()

    # ── MERFISH ───────────────────────────────────────────────────────
    merfish_paths = sorted([
        os.path.join(CLASS_DIR, f)
        for f in os.listdir(CLASS_DIR)
        if f.startswith('merfish_') and f.endswith('_classified.h5ad')
    ])

    merfish_methods = [
        ('cast_subclass', 'subclass', 'class'),
        ('hier_subclass', 'm3_subclass', 'm3_class'),
    ]

    process_dataset('merfish', merfish_paths, merfish_methods)

    # ── Slide-tags (Spearman) ─────────────────────────────────────────
    slidetags_paths = sorted([
        os.path.join(CLASS_DIR, f)
        for f in os.listdir(CLASS_DIR)
        if f.startswith('slidetags_') and f.endswith('_spearman.h5ad')
    ])

    slidetags_methods = [
        ('cast_subclass', 'subclass', 'class'),
        ('hier_subclass', 'm3_subclass', 'm3_class'),
    ]

    process_dataset('slidetags', slidetags_paths, slidetags_methods)

    # ── Xenium 5k ──────────────────────────────────────────────────────
    xenium_path = os.path.join(CLASS_DIR, 'xenium5k_annotated.h5ad')
    if os.path.exists(xenium_path):
        xenium_methods = [
            ('hier_subclass', 'hier_subclass', 'hier_class'),
        ]
        process_dataset('xenium5k', [xenium_path], xenium_methods,
                        condition_col='condition', animal_col='animal')

    print(f"\nTotal time: {time.time()-t0:.0f}s")
    print("Done!")


if __name__ == '__main__':
    main()
