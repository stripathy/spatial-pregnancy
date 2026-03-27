#!/usr/bin/env python3
"""
Benchmark cell type annotations against reference and CAST labels.

Computes:
  1. Agreement with CAST labels (class + subclass)
  2. Proportion correlation vs Zeng MERFISH reference
  3. Spatial coherence (k=20 nearest neighbor purity)
  4. Per-subclass spatial coherence comparison

Usage:
    python code/benchmark.py --input output/merfish_annotated.h5ad --ref-proportions output/data/ref_proportions.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import anndata as ad


WORKING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKING_DIR, 'code'))
from modules.correlation import spatial_coherence


def proportion_correlation(query_proportions, ref_proportions):
    """Pearson correlation between query and reference type proportions."""
    common = sorted(set(query_proportions.index) & set(ref_proportions.index))
    if len(common) < 3:
        return np.nan, len(common)
    q = query_proportions.reindex(common, fill_value=0)
    r = ref_proportions.reindex(common, fill_value=0)
    return np.corrcoef(q, r)[0, 1], len(common)


def run_benchmarks(adata, prefix='hier', ref_prop_path=None, sample_col='sample'):
    """Run all benchmarks on annotated adata."""

    results = []
    samples = sorted(adata.obs[sample_col].unique())
    coords = adata.obs[['x', 'y']].values

    # Load reference proportions
    ref_props = {}
    if ref_prop_path:
        for level in ['class', 'subclass', 'supertype']:
            p = ref_prop_path.replace('.csv', f'_{level}.csv')
            if os.path.exists(p):
                ref_props[level] = pd.read_csv(p, index_col=0).squeeze()

    for sample in samples:
        mask = adata.obs[sample_col] == sample
        obs = adata.obs[mask]
        n = mask.sum()
        bench = {'sample': sample, 'n_cells': n}

        # Agreement with CAST
        has_cast = 'class' in obs.columns and 'subclass' in obs.columns
        if has_cast:
            for level in ['class', 'subclass']:
                cast_col = level
                hier_col = f'{prefix}_{level}'
                if hier_col in obs.columns:
                    agree = (obs[cast_col].astype(str) == obs[hier_col].astype(str)).mean()
                    bench[f'{level}_agree'] = agree

        # Proportion correlation vs reference
        for level in ['class', 'subclass', 'supertype']:
            hier_col = f'{prefix}_{level}'
            if hier_col not in obs.columns:
                continue
            qp = obs[hier_col].value_counts(normalize=True)

            if level in ref_props:
                corr_val, n_common = proportion_correlation(qp, ref_props[level])
                bench[f'prop_corr_{level}'] = corr_val
                bench[f'prop_n_common_{level}'] = n_common

            # CAST proportions vs reference
            if has_cast and level in ['class', 'subclass'] and level in ref_props:
                cast_qp = obs[level].value_counts(normalize=True)
                cast_corr, _ = proportion_correlation(cast_qp, ref_props[level])
                bench[f'cast_prop_corr_{level}'] = cast_corr

        # Spatial coherence
        sample_coords = coords[mask.values]
        for level in ['class', 'subclass']:
            hier_col = f'{prefix}_{level}'
            if hier_col in obs.columns:
                coh = spatial_coherence(obs[hier_col].values, sample_coords)
                bench[f'spatial_{level}'] = coh.mean()

            if has_cast:
                cast_coh = spatial_coherence(obs[level].values, sample_coords)
                bench[f'cast_spatial_{level}'] = cast_coh.mean()

        results.append(bench)

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Benchmark cell type annotations')
    parser.add_argument('--input', required=True, help='Annotated h5ad file')
    parser.add_argument('--output', default=None, help='Output CSV for benchmarks')
    parser.add_argument('--prefix', default='hier', help='Annotation column prefix')
    parser.add_argument('--sample-col', default='sample')
    parser.add_argument('--ref-proportions', default=os.path.join(
        WORKING_DIR, 'output/classification_v2/ref'),
        help='Path prefix for reference proportion CSVs (e.g., output/ref -> ref_class.csv)')
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    adata = ad.read_h5ad(args.input)
    print(f"  {adata.shape[0]:,} cells, {adata.obs[args.sample_col].nunique()} samples")

    bench = run_benchmarks(adata, prefix=args.prefix,
                           ref_prop_path=args.ref_proportions,
                           sample_col=args.sample_col)

    print(f"\n{'='*70}")
    print("BENCHMARK RESULTS")
    print(f"{'='*70}")
    for _, row in bench.iterrows():
        parts = [f"{row['sample']:12s} ({row['n_cells']:,} cells)"]
        for col in ['class_agree', 'subclass_agree', 'prop_corr_class', 'spatial_class']:
            if col in row and pd.notna(row[col]):
                parts.append(f"{col}={row[col]:.3f}")
        print('  '.join(parts))

    # Means
    print(f"\nMeans:")
    for col in bench.columns:
        if col not in ['sample', 'n_cells'] and bench[col].dtype in [np.float64, np.float32]:
            print(f"  {col}: {bench[col].mean():.3f}")

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        bench.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
