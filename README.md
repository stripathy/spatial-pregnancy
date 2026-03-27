# spatial-pregnancy

Multi-platform spatial transcriptomic analysis of the pregnant mouse brain.

## Overview

This repo provides a complete pipeline for analyzing cell type composition and gene expression changes in the maternal mouse hypothalamus across pregnancy, using three spatial transcriptomics platforms:

| Platform | Technology | Gene panel | Animals (CTRL / PREG) |
|----------|-----------|------------|----------------------|
| MERFISH | Imaging-based in situ | 496 genes | 3 / 3 |
| Slide-tags | snRNA-seq + spatial barcoding | ~32,000 genes | 3 / 3 |
| Xenium 5k | In situ sequencing | 5,009 genes | 3 / 3 (2 sections each) |

All platforms are annotated against the **Allen Brain Cell Atlas** (ABCA) WMB-taxonomy, restricted to Zeng MERFISH reference sections (.46–.49): 24 classes, 135 subclasses, 502 supertypes.

## Pipeline

### Step 1: Reference data preparation

| Script | Description |
|--------|-------------|
| `01_download_zeng_atlas.py` | Download Zeng ABCA MERFISH reference metadata and expression |
| `01_run_mapmycells.py` | Run Allen Institute MapMyCells (HANN) for baseline annotations |
| `02_prep_zeng_atlas.py` | Prepare Zeng reference for CAST spatial alignment |
| `02_build_snrnaseq_centroids.py` | Build subclass-level centroids from ABCA precomputed stats |
| `03_subset_zeng_imputed.py` | Subset imputed expression to 4 target hypothalamic sections |
| `04_extract_zeng_reference.py` | Memory-efficient extraction of Zeng imputed reference |

### Step 2: Cell type annotation

| Script | Description |
|--------|-------------|
| `02_correlation_classifier.py` | Correlation-based classifier (Pearson for MERFISH/Xenium, Spearman for Slide-tags) |
| `03_two_pass_classifier.py` | Two-pass cross-modal classifier with confidence filtering |
| `03_classify_and_benchmark.py` | Full classification + benchmarking pipeline |
| `annotate.py` | Hierarchical annotation with region-restricted centroids |
| `benchmark.py` | Validation: CAST agreement, proportion correlation, spatial coherence |
| `convert_rds_to_h5ad.py` | Convert Seurat RDS objects to h5ad (for Xenium 5k data) |

### Step 3: Compositional analysis (crumblr)

| Script | Description |
|--------|-------------|
| `build_crumblr_input.py` | Build per-sample cell type count matrices |
| `run_crumblr.R` | Compositional testing with crumblr + dream (random effects for Xenium biological replicates) |

### Step 4: Differential expression

| Script | Description |
|--------|-------------|
| `run_de.py` | Pseudobulk DE with edgeR GLM quasi-likelihood F-tests |
| `meta_analyze_de.py` | Stouffer's weighted Z-score meta-analysis across platforms |

### Step 5: Visualization and reporting

| Script | Description |
|--------|-------------|
| `export_viewer.py` | Bundle annotated spatial data into a standalone HTML viewer |
| `viewer_template.html` | Interactive canvas-based spatial viewer (2.3M cells, 30 samples) |
| `plot_de_results.R` | Per-platform DE volcano and summary plots |
| `plot_meta_volcano.R` | Meta-analysis volcano plots |
| `plot_cross_platform_concordance.R` | Cross-platform logFC concordance plots |
| `plot_crumblr_forest.py` | Compositional analysis forest plots |
| `plot_de_concordant_hits.py` | Concordant DE forest plot and heatmap |
| `plot_de_robust_concordant.py` | Robust concordant DE figures (FDR<0.1 anchor + p<0.05 replicate) |
| `plot_analysis_summary_figures.py` | All figures for the analysis summary report |
| `write_analysis_summary.py` | Build analysis summary markdown from source data |
| `md_to_pdf.py` | Convert markdown report to PDF |

### Shared modules (`code/modules/`)

| Module | Description |
|--------|-------------|
| `config.py` | Pipeline constants (thresholds, seeds, gene panel sizes) |
| `correlation.py` | Correlation classifier, label assignment, spatial coherence |
| `gene_mapping.py` | Gene symbol to Ensembl ID mapping utilities |

## Key findings

### Annotation performance

| Dataset | Method | Class agreement | Subclass agreement | Proportion r (class) | Spatial coherence |
|---------|--------|----------------|-------------------|---------------------|-------------------|
| MERFISH (n=9) | Hierarchical Pearson | 65.2% | 47.4% | 0.861 | 0.345 |
| Slide-tags (n=8) | Hierarchical Spearman | 82.4% | 48.6% | 0.929 | 0.335 |
| Xenium 5k (n=12) | Hierarchical Pearson | — | — | 0.897 | — |

### Compositional changes

No cell types survive FDR correction in the cross-platform meta-analysis (n=3 animals per group). Suggestive per-platform hits (Xenium FDR<0.05) include Tanycytes, Monocytes, and NDB-SI-ant Prdm12 Gaba neurons.

### Differential expression

Cross-platform meta-analysis (Stouffer's method) of pseudobulk DE identifies 39 gene × cell type hits at FDR<0.10:

- **Prolactin (Prl)**: Upregulated (+3–4 logFC) in vascular/barrier cell types (astroependymal, VLMC, ABC, endothelial) — consistent with uptake of circulating maternal prolactin. Undetectable in Slide-tags (nuclear RNA misses cytoplasmic Prl mRNA).
- **GABAergic neuron regulation**: OT D3 Folh1 Gaba (10 downregulated genes) and LSX Nkx2-1 Gaba (4 downregulated genes) — lateral septum interneurons involved in social/reproductive behavior.
- **Apoe**: Downregulated across multiple GABAergic neuron types, suggesting altered lipid metabolism.
- **Cxcl13**: Upregulated in microglia, indicating immune surveillance changes at the blood-brain interface.

## Interactive viewer

A standalone HTML viewer for exploring all 2.3M cells across 30 samples:

```bash
python code/export_viewer.py
# Open output/viewer/pregnancy_viewer_standalone.html in a browser
```

Features: cell type filtering (neuronal/non-neuronal), subclass/class/confidence coloring, spatial tooltips with grid-based O(1) hit-testing, CAST label comparison, screenshot export.

## Data requirements

**Tracked in repo** (small reference files):
- `data/genename_mapping_merfish_panel.csv` — MERFISH gene panel mapping
- `data/reference/gene_symbol_to_ensembl_mouse.json` — general symbol → Ensembl mapping
- `data/reference/valid_celltypes_zeng_4sections.csv` — valid cell types from Zeng reference
- `data/XeniumPrimeMouse5Kpan_tissue_pathways_metadata.csv` — Xenium 5k gene panel metadata

**Not tracked** (download separately):
- `data/reference/precomputed_stats_ABC_revision_230821.h5` — [Allen Institute](https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/index.html#mapmycells/WMB-taxonomy/20240831/)
- Query h5ad files for each platform
- `output/data/adata_ref_zeng_imputed.h5ad` — generated by the reference preparation pipeline

## Installation

```bash
pip install -r requirements.txt
```

R dependencies (for crumblr and DE plotting):
```r
install.packages(c("BiocManager", "here", "ggplot2", "dplyr"))
BiocManager::install(c("crumblr", "dreamlet", "variancePartition", "limma", "edgeR"))
```
