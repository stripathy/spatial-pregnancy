#!/usr/bin/env python3
"""
Meta-analysis of pseudobulk DE results across MERFISH, Slide-tags, and Xenium 5k.

For each (gene, celltype) pair present in ≥2 platforms:
  - Stouffer's method to combine p-values (weighted by sqrt(n_samples))
  - Mean logFC (weighted by n_samples)
  - Cross-platform concordance (sign agreement)

Output: output/de/de_meta_PREG_vs_CTRL.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import stats

WORKING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKING_DIR, 'code'))
os.chdir(WORKING_DIR)

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--de-dir', default='output/de',
                    help='Directory containing per-platform DE CSV files')
_args, _ = parser.parse_known_args()

IN_DIR = _args.de_dir
OUT_DIR = _args.de_dir
os.makedirs(OUT_DIR, exist_ok=True)


def stouffer_combine(pvalues, weights):
    """Stouffer's weighted Z-score method."""
    pvalues = np.array(pvalues, dtype=float)
    weights = np.array(weights, dtype=float)
    # Clip p-values away from 0 and 1
    pvalues = np.clip(pvalues, 1e-300, 1 - 1e-15)
    z_scores = stats.norm.ppf(1 - pvalues)
    z_combined = np.dot(weights, z_scores) / np.sqrt(np.sum(weights**2))
    p_combined = 1 - stats.norm.cdf(z_combined)
    return z_combined, p_combined


def main():
    # Load DE results
    print("Loading DE results...")
    datasets = {
        'merfish':    ('output/de/de_merfish_PREG_vs_CTRL.csv',    3),   # n_animals per group
        'slidetags':  ('output/de/de_slidetags_PREG_vs_CTRL.csv',  3),
        'xenium5k':   ('output/de/de_xenium5k_PREG_vs_CTRL.csv',   3),
    }

    dfs = {}
    for name, (path, n_animals) in datasets.items():
        if not os.path.exists(path):
            print(f"  Missing: {path}, skipping")
            continue
        df = pd.read_csv(path)
        df['platform'] = name
        df['n_animals'] = n_animals
        df['weight'] = np.sqrt(n_animals)
        dfs[name] = df
        print(f"  {name}: {len(df):,} rows, {df['celltype'].nunique()} cell types")

    if len(dfs) < 2:
        print("Need at least 2 platforms, exiting.")
        return

    combined = pd.concat(dfs.values(), ignore_index=True)

    # ── Gene-level meta-analysis per (gene, celltype) ─────────────────────
    print("\nRunning gene-level meta-analysis...")

    results = []
    groups = combined.groupby(['gene', 'celltype'])
    n_groups = len(groups)

    for i, ((gene, celltype), grp) in enumerate(groups):
        if i % 50000 == 0:
            print(f"  {i:,}/{n_groups:,}...")

        n_platforms = grp['platform'].nunique()
        if n_platforms < 2:
            continue

        pvals = grp['PValue'].values
        logfcs = grp['logFC'].values
        weights = grp['weight'].values
        platforms = grp['platform'].values

        z_comb, p_comb = stouffer_combine(pvals, weights)

        # Weighted mean logFC
        mean_logfc = np.average(logfcs, weights=weights)

        # Sign concordance
        signs = np.sign(logfcs)
        dominant_sign = np.sign(mean_logfc)
        n_agree = (signs == dominant_sign).sum()
        concordant = n_agree == n_platforms

        row = {
            'gene': gene,
            'celltype': celltype,
            'z_combined': z_comb,
            'p_combined': p_comb,
            'mean_logFC': mean_logfc,
            'n_platforms': n_platforms,
            'platforms': ','.join(sorted(platforms)),
            'n_agree': n_agree,
            'concordant': concordant,
        }

        # Per-platform logFC
        for name in ['merfish', 'slidetags', 'xenium5k']:
            sub = grp[grp['platform'] == name]
            if len(sub) > 0:
                row[f'logFC_{name}'] = sub['logFC'].iloc[0]
                row[f'pval_{name}'] = sub['PValue'].iloc[0]
                row[f'fdr_{name}'] = sub['FDR'].iloc[0]
            else:
                row[f'logFC_{name}'] = np.nan
                row[f'pval_{name}'] = np.nan
                row[f'fdr_{name}'] = np.nan

        results.append(row)

    meta_df = pd.DataFrame(results)
    print(f"  {len(meta_df):,} (gene, celltype) pairs in ≥2 platforms")

    # FDR correction
    from statsmodels.stats.multitest import multipletests
    _, fdr, _, _ = multipletests(meta_df['p_combined'], method='fdr_bh')
    meta_df['fdr_combined'] = fdr

    # Sort by p_combined
    meta_df = meta_df.sort_values('p_combined').reset_index(drop=True)

    # Save
    outpath = os.path.join(OUT_DIR, 'de_meta_PREG_vs_CTRL.csv')
    meta_df.to_csv(outpath, index=False)
    print(f"\nSaved: {outpath} ({len(meta_df):,} rows)")

    # Summary
    n_fdr05 = (meta_df['fdr_combined'] < 0.05).sum()
    n_fdr10 = (meta_df['fdr_combined'] < 0.10).sum()
    n_nom05 = (meta_df['p_combined'] < 0.05).sum()
    print(f"\nSummary (≥2 platforms):")
    print(f"  FDR<0.05: {n_fdr05:,}")
    print(f"  FDR<0.10: {n_fdr10:,}")
    print(f"  nominal p<0.05: {n_nom05:,}")

    # ── Per-celltype concordance (Slide-tags vs Xenium) ────────────────────
    print("\nSlide-tags vs Xenium 5k concordance per cell type:")
    both = meta_df[meta_df['platforms'].str.contains('slidetags') &
                   meta_df['platforms'].str.contains('xenium5k')].copy()
    both = both.dropna(subset=['logFC_slidetags', 'logFC_xenium5k'])

    per_ct = []
    for ct, g in both.groupby('celltype'):
        if len(g) < 20:
            continue
        r, p = stats.pearsonr(g['logFC_slidetags'], g['logFC_xenium5k'])
        rho, _ = stats.spearmanr(g['logFC_slidetags'], g['logFC_xenium5k'])
        per_ct.append({'celltype': ct, 'pearson_r': r, 'spearman_rho': rho,
                       'n_genes': len(g), 'pearson_p': p})

    per_ct_df = pd.DataFrame(per_ct).sort_values('pearson_r', ascending=False)
    per_ct_df.to_csv(os.path.join(OUT_DIR, 'de_concordance_slidetags_xenium_per_ct.csv'), index=False)

    print(f"\nTop concordant cell types (ST vs Xenium):")
    print(per_ct_df.head(10).to_string(index=False))

    # All-gene concordance
    all_both = both.dropna(subset=['logFC_slidetags', 'logFC_xenium5k'])
    r_all, p_all = stats.pearsonr(all_both['logFC_slidetags'], all_both['logFC_xenium5k'])
    print(f"\nOverall ST vs Xenium r={r_all:.3f} (n={len(all_both):,} pairs)")

    print("\nDone!")


if __name__ == '__main__':
    main()
