#!/usr/bin/env python3
"""
Pseudobulk differential expression analysis using edgepython.

For each cell type (subclass), aggregates raw counts per sample,
then runs edgeR GLM quasi-likelihood test for condition effects.

Datasets: MERFISH, Slide-tags, Xenium 5k
Contrast: PREG vs CTRL (all datasets), POSTPART vs CTRL (MERFISH/Slide-tags only)
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import edgepython as ep

WORKING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKING_DIR, 'code'))
os.chdir(WORKING_DIR)

from modules.config import MIN_CELLS_PER_PB, MIN_SAMPLES_PER_GROUP, MIN_GENES_AFTER_FILTER

OUT_DIR = 'output/de'
os.makedirs(OUT_DIR, exist_ok=True)


def log(msg):
    print(msg, flush=True)


def build_pseudobulk(adata, celltype_col, sample_col, gene_names=None):
    """Sum raw counts per (sample, celltype) -> dict of {celltype: (counts_matrix, sample_info)}

    Returns dict: celltype -> {
        'counts': np.array (genes x samples),
        'gene_names': list,
        'sample_info': pd.DataFrame with sample, condition, [animal] columns,
        'n_cells': list of cell counts per sample,
    }
    """
    if gene_names is None:
        gene_names = list(adata.var_names)

    obs = adata.obs
    X = adata.X

    celltypes = sorted(obs[celltype_col].unique())
    results = {}

    for ct in celltypes:
        ct_mask = obs[celltype_col] == ct
        ct_obs = obs[ct_mask]
        ct_X = X[ct_mask.values]

        samples = sorted(ct_obs[sample_col].unique())
        sample_counts = []
        sample_info_rows = []
        n_cells_list = []

        for sid in samples:
            s_mask = ct_obs[sample_col] == sid
            n_cells = s_mask.sum()
            if n_cells < MIN_CELLS_PER_PB:
                continue

            s_X = ct_X[s_mask.values]
            if sp.issparse(s_X):
                pb = np.asarray(s_X.sum(axis=0)).ravel()
            else:
                pb = s_X.sum(axis=0).ravel()

            sample_counts.append(pb)
            n_cells_list.append(n_cells)

            # Get metadata for this sample
            row = {'sample': sid}
            s_obs = ct_obs[s_mask]
            if 'condition' in s_obs.columns:
                row['condition'] = s_obs['condition'].iloc[0]
            if 'animal' in s_obs.columns:
                row['animal'] = s_obs['animal'].iloc[0]
            sample_info_rows.append(row)

        if len(sample_counts) < 3:
            continue

        counts_matrix = np.column_stack(sample_counts).astype(np.float64)  # genes x samples
        sample_info = pd.DataFrame(sample_info_rows)

        results[ct] = {
            'counts': counts_matrix,
            'gene_names': gene_names,
            'sample_info': sample_info,
            'n_cells': n_cells_list,
        }

    return results


def run_de_for_celltype(ct_name, ct_data, contrast_col='condition',
                        ref_level='CTRL', test_level='PREG'):
    """Run edgeR DE for one cell type, one contrast."""

    counts = ct_data['counts']
    gene_names = ct_data['gene_names']
    info = ct_data['sample_info']

    # Filter to samples in the two groups
    mask = info[contrast_col].isin([ref_level, test_level])
    if mask.sum() < 3:
        return None

    info_sub = info[mask].reset_index(drop=True)
    counts_sub = counts[:, mask.values]

    n_ref = (info_sub[contrast_col] == ref_level).sum()
    n_test = (info_sub[contrast_col] == test_level).sum()

    if n_ref < MIN_SAMPLES_PER_GROUP or n_test < MIN_SAMPLES_PER_GROUP:
        return None

    # Design matrix
    design = pd.DataFrame({
        'Intercept': np.ones(len(info_sub)),
        test_level: (info_sub[contrast_col] == test_level).astype(float),
    })

    # Create DGEList
    genes_df = pd.DataFrame({'gene': gene_names})
    samples_df = info_sub[['sample']].copy()
    samples_df['group'] = info_sub[contrast_col].values

    try:
        dge = ep.make_dgelist(counts=counts_sub, genes=genes_df, samples=samples_df)

        # Filter low-expressed genes
        keep = ep.filter_by_expr(dge, design=design)
        if keep.sum() < MIN_GENES_AFTER_FILTER:
            return None

        counts_filt = counts_sub[keep]
        genes_filt = genes_df[keep].reset_index(drop=True)

        dge = ep.make_dgelist(counts=counts_filt, genes=genes_filt, samples=samples_df)
        dge = ep.calc_norm_factors(dge, method='TMM')
        dge = ep.estimate_disp(dge, design=design, robust=True)

        fit = ep.glm_ql_fit(dge, design=design, robust=True)
        results = ep.glm_ql_ftest(fit, coef=1)  # test_level coefficient

        tt = ep.top_tags(results, n=keep.sum(), sort_by='PValue')

        if isinstance(tt, dict) and 'table' in tt:
            df = tt['table'].copy()
        elif hasattr(tt, 'table'):
            df = tt.table.copy()
        elif isinstance(tt, pd.DataFrame):
            df = tt.copy()
        else:
            df = pd.DataFrame(tt)

        df['celltype'] = ct_name
        df['contrast'] = f'{test_level}_vs_{ref_level}'
        df['n_ref'] = n_ref
        df['n_test'] = n_test
        df['n_genes_tested'] = keep.sum()

        return df

    except Exception as e:
        log(f"    Error in {ct_name}: {e}")
        return None


def process_dataset(name, adata, celltype_col, sample_col, contrasts):
    """Run DE for all cell types and contrasts in one dataset."""

    log(f"\n{'='*70}")
    log(f"DE Analysis: {name}")
    log(f"{'='*70}")
    log(f"  {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")
    log(f"  Cell types: {adata.obs[celltype_col].nunique()}")
    log(f"  Samples: {adata.obs[sample_col].nunique()}")

    # Build pseudobulk
    log("  Building pseudobulk...")
    t0 = time.time()
    pb = build_pseudobulk(adata, celltype_col, sample_col)
    log(f"  Built {len(pb)} cell types in {time.time()-t0:.0f}s")

    all_results = []

    for ref_level, test_level in contrasts:
        log(f"\n  --- Contrast: {test_level} vs {ref_level} ---")
        contrast_results = []

        for ct_name in sorted(pb.keys()):
            df = run_de_for_celltype(ct_name, pb[ct_name],
                                      ref_level=ref_level, test_level=test_level)
            if df is not None:
                n_sig = (df['FDR'] < 0.05).sum() if 'FDR' in df.columns else 0
                n_nom = (df['PValue'] < 0.05).sum() if 'PValue' in df.columns else 0
                log(f"    {ct_name:40s}: {len(df):5d} genes, {n_sig:3d} FDR<0.05, {n_nom:3d} nom<0.05")
                contrast_results.append(df)

        if contrast_results:
            combined = pd.concat(contrast_results, ignore_index=True)
            combined['dataset'] = name
            outpath = f'{OUT_DIR}/de_{name}_{test_level}_vs_{ref_level}.csv'
            combined.to_csv(outpath, index=False)
            log(f"  Saved {outpath}: {len(combined):,} rows, "
                f"{len(contrast_results)} cell types")
            all_results.append(combined)

    return all_results


def main():
    t0 = time.time()

    # ── MERFISH ───────────────────────────────────────────────────────
    log("Loading MERFISH...")
    merfish = ad.read_h5ad('data/merfish_clean/merfish_normalized/adata_query_merfish_pub.h5ad')
    # Convert categoricals to string to avoid merge issues
    for col in merfish.obs.columns:
        if merfish.obs[col].dtype.name == 'category':
            merfish.obs[col] = merfish.obs[col].astype(str)
    # Add hier annotations
    merfish.obs['m3_class'] = 'Unknown'
    merfish.obs['m3_subclass'] = 'Unknown'
    for f in sorted(os.listdir('output/classification_v2')):
        if f.startswith('merfish_') and f.endswith('_classified.h5ad'):
            a = ad.read_h5ad(f'output/classification_v2/{f}', backed='r')
            sample = a.obs['sample'].iloc[0]
            mask = merfish.obs['sample'] == sample
            for col in ['m3_class', 'm3_subclass']:
                if col in a.obs.columns:
                    merfish.obs.loc[mask, col] = a.obs[col].astype(str).values
            a.file.close()

    merfish_results = process_dataset(
        'merfish', merfish, 'm3_subclass', 'sample',
        contrasts=[('CTRL', 'PREG'), ('CTRL', 'POSTPART')]
    )
    del merfish

    # ── Slide-tags ────────────────────────────────────────────────────
    log("\nLoading Slide-tags...")
    slidetags = ad.read_h5ad('data/GSE313279_slide_tags.h5ad')
    for col in slidetags.obs.columns:
        if slidetags.obs[col].dtype.name == 'category':
            slidetags.obs[col] = slidetags.obs[col].astype(str)
    slidetags.obs['m3_class'] = 'Unknown'
    slidetags.obs['m3_subclass'] = 'Unknown'
    for f in sorted(os.listdir('output/classification_v2')):
        if f.startswith('slidetags_') and f.endswith('_spearman.h5ad'):
            a = ad.read_h5ad(f'output/classification_v2/{f}', backed='r')
            sample_val = a.obs['sample'].iloc[0]
            mask = slidetags.obs['sample'] == sample_val
            for col in ['m3_class', 'm3_subclass']:
                if col in a.obs.columns:
                    slidetags.obs.loc[mask, col] = a.obs[col].astype(str).values
            a.file.close()

    slidetags.obs['condition'] = slidetags.obs['sample'].astype(str).str.extract(r'^(CTRL|PREG|POSTPART)')[0].values

    slidetags_results = process_dataset(
        'slidetags', slidetags, 'm3_subclass', 'sample',
        contrasts=[('CTRL', 'PREG'), ('CTRL', 'POSTPART')]
    )
    del slidetags

    # ── Xenium 5k ─────────────────────────────────────────────────────
    log("\nLoading Xenium 5k...")
    xenium = ad.read_h5ad('output/classification_v2/xenium5k_annotated.h5ad')

    # Use animal as sample unit for pseudobulk (biological replicates)
    xenium_results = process_dataset(
        'xenium5k', xenium, 'hier_subclass', 'animal',
        contrasts=[('CTRL', 'PREG')]
    )
    del xenium

    log(f"\n{'='*70}")
    log(f"Total time: {time.time()-t0:.0f}s")
    log("Done!")


if __name__ == '__main__':
    main()
