# Xenium 5k Dataset Metadata

## Overview

- **Technology:** 10x Xenium Prime 5K pan-tissue pathways panel (mouse)
- **Panel:** 5,006 genes (+ 3 control probes: Efna4, Gcamp6f, mCherry)
- **Species:** Mouse (Mus musculus)
- **Brain region:** Coronal sections (similar anteroposterior level to MERFISH/Slide-tags)
- **Total cells:** 1,271,700
- **Conditions:** Virgin (control) and Pregnant (E18)

## Sample Structure

Each animal contributes **two adjacent coronal sections** on the same Xenium slide. Adjacent sections from the same animal are **technical replicates**, not independent biological replicates. This was confirmed by proportion correlation analysis: tech rep pairs have mean r = 0.993 vs biological replicate mean r = 0.961 (p = 0.001).

| Section | Animal | Condition | Cells |
|---------|--------|-----------|-------|
| virgin1 | virgin_1 | CTRL | 73,839 |
| virgin2 | virgin_1 | CTRL | 75,872 |
| virgin3 | virgin_2 | CTRL | 116,583 |
| virgin4 | virgin_2 | CTRL | 120,053 |
| virgin5 | virgin_3 | CTRL | 121,836 |
| virgin6 | virgin_3 | CTRL | 116,482 |
| preg1 | preg_1 | PREG | 52,333 |
| preg2 | preg_1 | PREG | 43,331 |
| preg3 | preg_2 | PREG | 140,558 |
| preg4 | preg_2 | PREG | 133,894 |
| preg5 | preg_3 | PREG | 132,962 |
| preg6 | preg_3 | PREG | 143,957 |

**Effective biological n:** 3 virgin, 3 pregnant (6 animals total, 12 sections)

## Statistical Modeling

For compositional or differential expression analyses, models must account for the nested structure:

- **Fixed effect:** `condition` (CTRL vs PREG)
- **Random effect:** `animal` (to account for within-animal correlation between adjacent sections)
- **Sample-level unit:** `sample` (section ID) for pseudobulk aggregation

Example crumblr formula: `~ condition + (1|animal)`

## Notes

- `preg1`/`preg2` are smaller sections (52K and 43K cells) compared to all others (>73K), likely from a less ideal cutting plane
- The h5ad file contains columns: `sample` (section ID), `condition` (CTRL/PREG), `animal` (biological replicate ID)
- Gene panel metadata with Ensembl IDs: `data/XeniumPrimeMouse5Kpan_tissue_pathways_metadata.csv`
- All 5,006 panel genes have 100% coverage in the Allen precomputed stats reference
