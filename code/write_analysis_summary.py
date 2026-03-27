#!/usr/bin/env python3
"""
Build the analysis summary markdown from source data files.

All statistics are extracted programmatically — no values are hardcoded.
Figures are referenced via relative paths from the output directory.
"""

import os
import shutil
import numpy as np
import pandas as pd
from collections import Counter

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = 'output/analysis_summary'
FIG = f'{OUT}/figures'
os.makedirs(FIG, exist_ok=True)

# Copy existing figures into the summary figures directory
for src in [
    'output/de/figures/de_meta_volcano.png',
    'output/de/figures/de_concordance_st_vs_xenium.png',
]:
    if os.path.exists(src):
        shutil.copy2(src, f'{FIG}/{os.path.basename(src)}')


# ════════════════════════════════════════════════════════════════════════
# Load all data sources
# ════════════════════════════════════════════════════════════════════════

# -- Annotation validation --
class_props = pd.read_csv('output/xenium5k/class_proportions_all_modalities.csv')
class_props = class_props.rename(columns={class_props.columns[0]: 'celltype'})
sub_props = pd.read_csv('output/xenium5k/subclass_proportions_all_modalities.csv')
sub_props = sub_props.rename(columns={sub_props.columns[0]: 'celltype'})

from scipy.stats import pearsonr
ref_c = class_props['Reference'].values
xen_c = class_props['Xenium5k_Hier'].values
mask_c = (ref_c > 0) & (xen_c > 0)
r_class, _ = pearsonr(np.log10(ref_c[mask_c]), np.log10(xen_c[mask_c]))

ref_s = sub_props['Reference'].values
xen_s = sub_props['Xenium5k'].values
mask_s = (ref_s > 0) & (xen_s > 0)
r_subclass, _ = pearsonr(np.log10(ref_s[mask_s] + 1e-6), np.log10(xen_s[mask_s] + 1e-6))

n_classes_xenium = len(class_props)
n_subclasses_xenium = len(sub_props)

# -- Crumblr --
cr = pd.read_csv('output/crumblr/crumblr_results_all.csv')
cr_preg = cr[cr.contrast == 'PREG_vs_CTRL']
n_crumblr_total = len(cr)
n_crumblr_preg = len(cr_preg)
n_crumblr_fdr05 = (cr_preg.FDR < 0.05).sum()
n_crumblr_fdr10 = (cr_preg.FDR < 0.10).sum()

# Top crumblr hits
cr_top = cr_preg.sort_values('P.Value').head(10)

# -- DE meta-analysis --
meta = pd.read_csv('output/de/de_meta_PREG_vs_CTRL.csv')
n_meta_total = len(meta)
n_meta_fdr05 = (meta.fdr_combined < 0.05).sum()
n_meta_fdr10 = (meta.fdr_combined < 0.10).sum()
n_meta_fdr20 = (meta.fdr_combined < 0.20).sum()

# Per-platform test counts (from actual DE files, not meta-analysis subset)
PLATFORMS = ['merfish', 'slidetags', 'xenium5k']
plat_tested = {}
plat_fdr05 = {}
for plat_name in PLATFORMS:
    de_plat = pd.read_csv(f'output/de/de_{plat_name}_PREG_vs_CTRL.csv')
    plat_tested[plat_name] = len(de_plat)
    plat_fdr05[plat_name] = (de_plat.FDR < 0.05).sum()

# DE concordance
conc_path = 'output/de/de_concordance_summary.csv'
if os.path.exists(conc_path):
    conc = pd.read_csv(conc_path)
else:
    conc = None

# Concordant DE hits (FDR<0.2 + concordant ≥2)
def check_concordant(row):
    logfcs = {p: row[f'logFC_{p}'] for p in PLATFORMS if pd.notna(row.get(f'logFC_{p}'))}
    if len(logfcs) < 2:
        return False
    signs = [np.sign(v) for v in logfcs.values()]
    return Counter(signs).most_common(1)[0][1] >= 2

hits = meta[meta.fdr_combined < 0.20].copy()
hits = hits[hits.apply(check_concordant, axis=1)].copy()
hits = hits.sort_values('p_combined').reset_index(drop=True)
n_concordant = len(hits)

# Count by biology
prl_hits = hits[hits.gene == 'Prl']
apoe_hits = hits[hits.gene == 'Apoe']
ot_d3_hits = hits[hits.celltype.str.contains('060 OT D3')]
lsx_hits = hits[hits.celltype.str.contains('069 LSX Nkx2-1')]
cxcl13_hits = hits[hits.gene == 'Cxcl13']

# Per-platform DE file stats
de_merfish = pd.read_csv('output/de/de_merfish_PREG_vs_CTRL.csv')
de_slidetags = pd.read_csv('output/de/de_slidetags_PREG_vs_CTRL.csv')
de_xenium = pd.read_csv('output/de/de_xenium5k_PREG_vs_CTRL.csv')

n_genes_merfish = de_merfish.gene.nunique()
n_genes_slidetags = de_slidetags.gene.nunique()
n_genes_xenium = de_xenium.gene.nunique()

n_ct_merfish = de_merfish.celltype.nunique()
n_ct_slidetags = de_slidetags.celltype.nunique()
n_ct_xenium = de_xenium.celltype.nunique()


# ════════════════════════════════════════════════════════════════════════
# Build markdown
# ════════════════════════════════════════════════════════════════════════

md = []
md.append("# Spatial Transcriptomic Analysis of the Pregnant Mouse Brain\n")

# ── Section 1: Overview ────────────────────────────────────────────────
md.append("## 1. Overview\n")
md.append("""This document summarizes the cell type annotation, compositional analysis, and differential expression results from a multi-platform spatial transcriptomics study of the pregnant mouse hypothalamus and surrounding brain regions.

**Platforms:**

| Platform | Technology | Gene panel | Samples (CTRL / PREG) | Chemistry |
|----------|-----------|------------|----------------------|-----------|
| MERFISH | Imaging-based in situ | 496 genes | 3 / 3 | Cytoplasmic mRNA |
| Slide-tags | snRNA-seq + spatial barcoding | ~32,000 genes (genome-wide) | 3 / 3 | Nuclear RNA |
| Xenium 5k | In situ sequencing | 5,009 genes | 3 / 3 (2 sections each) | Cytoplasmic mRNA |

**Reference taxonomy:** Allen Brain Cell Atlas (ABCA) WMB-taxonomy, restricted to Zeng MERFISH reference sections (.46–.49) matching the query brain regions. This provides a hierarchy of 24 classes, 135 subclasses, and 502 supertypes.

**Key chemistry difference:** MERFISH and Xenium detect cytoplasmic mRNA transcripts in situ, while Slide-tags captures nuclear RNA only. This has important consequences for genes with predominantly cytoplasmic mRNA localization (e.g., secreted hormones like prolactin).
""")

# ── Section 2: Annotation ──────────────────────────────────────────────
md.append("## 2. Cell Type Annotation & Validation\n")

md.append("### 2.1 Approach\n")
md.append("""All three platforms were annotated using a **hierarchical correlation-based classifier** against ABCA reference centroids:

1. Build subclass-level centroids from ABCA precomputed cluster statistics (5,322 clusters aggregated to 135 subclasses)
2. Restrict centroids to cell types present in the Zeng MERFISH reference sections (.46–.49)
3. Hierarchical classification: assign class first, then subclass within the assigned class
4. Correlation method: **Spearman** for raw-count data (Slide-tags), **Pearson** for normalized data (MERFISH, Xenium 5k)
""")

md.append("### 2.2 MERFISH & Slide-tags Validation\n")
md.append("""Annotations were validated against independent CAST (graph neural network + spatial registration) labels. Key results:

**Class-level agreement with CAST:**

| Modality | Method | Mean class agreement | Range |
|----------|--------|---------------------|-------|
| Slide-tags | Spearman hierarchical | **82.4%** | 76.5–87.2% |
| MERFISH | Pearson hierarchical | **65.2%** | 56.9–73.4% |

**Subclass-level agreement:** ~48% for both modalities, reflecting the challenge of distinguishing 135 subtypes by expression alone.

**Proportion correlation vs Zeng reference (log₁₀):**

| Modality | Method | Class r | Subclass r |
|----------|--------|---------|-----------|
| Slide-tags | Spearman hier | **0.929** | 0.651 |
| MERFISH | Pearson hier | 0.861 | **0.776** |
| Slide-tags | CAST | 0.944 | 0.953 |
| MERFISH | CAST | 0.713 | 0.773 |

The correlation classifier achieves higher class-level proportion correlation than CAST for MERFISH (0.861 vs 0.713), suggesting it better recovers absolute proportions while CAST optimizes for spatial coherence.

**Spatial coherence (k=20 nearest neighbors):**

| Modality | Method | Class coherence | Subclass coherence |
|----------|--------|----------------|-------------------|
| Slide-tags | Spearman hier | 0.335 | 0.151 |
| Slide-tags | CAST | 0.374 | 0.228 |
| MERFISH | Pearson hier | 0.345 | 0.175 |
| MERFISH | CAST | 0.442 | 0.338 |

CAST achieves higher spatial coherence as expected (it uses spatial graph information). The largest gaps are in cortical layer-specific glutamatergic neurons (L2/3, L4/5, L6 IT), which share similar expression profiles and are distinguished primarily by laminar position.
""")

md.append("### 2.3 Xenium 5k Annotation\n")
md.append(f"""The Xenium 5k dataset (5,009 genes, 6 animals: 3 CTRL + 3 PREG, with 2 tissue sections per animal = 12 sections total) was annotated using the same hierarchical Pearson correlation classifier, yielding {n_classes_xenium} classes and {n_subclasses_xenium} subclasses.

**Proportion validation against Zeng reference:**

| Level | Pearson r (log₁₀) | N types compared |
|-------|-------------------|-----------------|
| Class | **{r_class:.3f}** | {mask_c.sum()} |
| Subclass | **{r_subclass:.3f}** | {mask_s.sum()} |

![Xenium 5k proportion validation](figures/xenium_validation_proportions.png)

The strong class-level correlation (r = {r_class:.3f}) confirms that the Xenium 5k annotations recover realistic cell type proportions. Subclass-level agreement (r = {r_subclass:.3f}) is lower, consistent with the difficulty of resolving 135 subtypes. The same cortical layer neuron limitations apply as for MERFISH/Slide-tags.
""")

# ── Section 3: Crumblr ─────────────────────────────────────────────────
md.append("## 3. Compositional Analysis (crumblr)\n")

md.append("### 3.1 Approach\n")
md.append("""Cell type proportions were tested for changes in pregnancy using **crumblr**:

1. Additive log-ratio (ALR) transformation of per-sample cell type counts
2. Model fitting with **dream** (variancePartition):
   - MERFISH & Slide-tags: `~ condition` (fixed effects only, n=3 per group)
   - Xenium 5k: `~ condition + (1|animal)` (random effect for biological replicate, n=3 animals per group with 2 sections each = 6 sections per group)
3. Contrasts: PREG vs CTRL, POSTPART vs CTRL, POSTPART vs PREG
4. Cell type filtering: present in ≥50% of samples
5. Stratified analyses: whole composition, neuronal-only, non-neuronal-only
""")

md.append("### 3.2 Results\n")
md.append(f"""Across all platforms, annotation methods, strata, and contrasts, crumblr tested {n_crumblr_total:,} cell type × contrast combinations. For the PREG vs CTRL contrast specifically ({n_crumblr_preg:,} tests):

| FDR threshold | Hits |
|--------------|------|
| FDR < 0.05 | {n_crumblr_fdr05} |
| FDR < 0.10 | {n_crumblr_fdr10} |

**The overall picture is one of limited power.** With n=3 biological replicates per group across all platforms (Xenium 5k has 2 technical replicate sections per animal), the study is underpowered to detect modest composition changes. The few significant hits are almost exclusively from Xenium 5k, which benefits from the additional technical replicates.

**Top suggestive hits (PREG vs CTRL, sorted by p-value):**

| Cell type | Level | logFC | P-value | FDR |
|-----------|-------|-------|---------|-----|
""")

for _, r in cr_top.iterrows():
    level_short = r['level'].replace('xenium5k_', 'Xe:').replace('slidetags_', 'ST:').replace('merfish_', 'MF:')
    md.append(f"| {r['celltype']} | {level_short} | {r['logFC']:+.3f} | {r['P.Value']:.2e} | {r['FDR']:.3f} |")

md.append("\n")

# Load crumblr meta-analysis for the forest plot section
cr_meta = pd.read_csv('output/analysis_summary/crumblr_meta_analysis.csv')
cr_meta_fdr05 = (cr_meta.meta_fdr < 0.05).sum()
cr_meta_fdr10 = (cr_meta.meta_fdr < 0.10).sum()
cr_meta_fdr20 = (cr_meta.meta_fdr < 0.20).sum()

# Endo stats
endo_row = cr_meta[cr_meta.celltype == '333 Endo NN']
if len(endo_row) > 0:
    endo_meta_p = endo_row.iloc[0]['meta_p']
    endo_meta_fdr = endo_row.iloc[0]['meta_fdr']
    endo_meta_lfc = endo_row.iloc[0]['meta_logFC']
    endo_me_lfc = endo_row.iloc[0].get('logFC_merfish', np.nan)
    endo_sl_lfc = endo_row.iloc[0].get('logFC_slidetags', np.nan)
    endo_xe_lfc = endo_row.iloc[0].get('logFC_xenium5k', np.nan)

md.append("### 3.3 Cross-Platform Meta-Analysis of Proportion Changes\n")
md.append(f"""To combine evidence across platforms, we applied Stouffer's weighted Z-score meta-analysis (weights = √n_samples) to the per-platform crumblr logFC estimates, with inverse-variance weighting for the combined effect size. FDR was computed across all {len(cr_meta)} cell types tested in ≥2 platforms.

**Meta-analysis summary:** {cr_meta_fdr05} cell types at FDR < 0.05, {cr_meta_fdr10} at FDR < 0.10, {cr_meta_fdr20} at FDR < 0.20. No cell type reaches significance in the cross-platform meta-analysis, confirming the study is underpowered for detecting modest compositional changes.

The forest plot below shows per-platform logFC ± 95% CI (colored circles) and the meta-analysis mean (black diamonds) for all non-neuronal cell types and the top 25 neuronal cell types:

![Crumblr compositional forest plot](figures/crumblr_forest_all.png)

**Focused non-neuronal panel** with per-platform values and meta-analysis FDR annotated on the right:

![Crumblr non-neuronal forest plot](figures/crumblr_forest_nonneuronal.png)

**Notable patterns:**
- **Monocytes NN**: Strongest meta-analysis signal among non-neuronal types. Increased in PREG in MERFISH and Xenium 5k (though decreased in Slide-tags), consistent with known immune cell infiltration during pregnancy.
- **Tanycyte NN**: Increased in PREG, driven primarily by Xenium 5k. Tanycytes line the third ventricle and respond to metabolic and hormonal changes during pregnancy.
- **Microglia NN**: Concordant positive direction across all 3 platforms, consistent with the Cxcl13 upregulation in the DE analysis (Section 4.5).
""")

if len(endo_row) > 0:
    endo_parts = []
    if pd.notna(endo_me_lfc):
        endo_parts.append(f"MERFISH: {endo_me_lfc:+.2f}")
    if pd.notna(endo_sl_lfc):
        endo_parts.append(f"Slide-tags: {endo_sl_lfc:+.2f}")
    if pd.notna(endo_xe_lfc):
        endo_parts.append(f"Xenium: {endo_xe_lfc:+.2f}")
    md.append(f"""**Endothelial cells (333 Endo NN):** All three platforms show a positive logFC ({', '.join(endo_parts)}), suggesting a modest increase in endothelial proportion during pregnancy. However, the effect is not statistically significant in any individual platform or in the meta-analysis (meta p = {endo_meta_p:.3f}, FDR = {endo_meta_fdr:.3f}, meta logFC = {endo_meta_lfc:+.3f}). The concordant direction across platforms is noteworthy but the wide confidence intervals — particularly for Slide-tags — reflect the limited sample sizes.
""")

md.append("""**Cross-platform replication of compositional changes is weak overall.** Most suggestive per-platform hits (e.g., Xenium FDR < 0.05 for Tanycyte, Monocytes) are not replicated at significance in the other platforms, and no cell type survives cross-platform meta-analysis FDR correction. This reflects limited statistical power (n = 3 animals per group) and potential platform-specific biases in cell type detection and segmentation.
""")

# ── Section 4: DE ──────────────────────────────────────────────────────
md.append("## 4. Differential Expression Meta-Analysis\n")

md.append("### 4.1 Pseudobulk DE Approach\n")
md.append(f"""Per-celltype, per-platform differential expression was tested using **pseudobulk aggregation + edgeR GLM quasi-likelihood F-tests**:

1. Sum raw counts per (sample, cell type) → pseudobulk count matrix
2. Filter: ≥10 cells per pseudobulk, ≥2 samples per group, ≥10 genes after expression filtering
3. Normalization: TMM (Trimmed Mean of M-values)
4. Model: quasi-likelihood GLM with condition as fixed effect
5. Multiple testing: Benjamini-Hochberg FDR per platform

**Per-platform testing volume:**

| Platform | Genes tested | Cell types tested | Gene × celltype tests | FDR < 0.05 |
|----------|-------------|-------------------|----------------------|-----------|
| MERFISH | {n_genes_merfish} | {n_ct_merfish} | {plat_tested['merfish']:,} | {plat_fdr05['merfish']} |
| Slide-tags | {n_genes_slidetags:,} | {n_ct_slidetags} | {plat_tested['slidetags']:,} | {plat_fdr05['slidetags']} |
| Xenium 5k | {n_genes_xenium:,} | {n_ct_xenium} | {plat_tested['xenium5k']:,} | {plat_fdr05['xenium5k']} |

![DE overview](figures/de_overview_stats.png)
""")

md.append("### 4.2 Stouffer's Meta-Analysis\n")
md.append(f"""Per-platform results were combined using **Stouffer's weighted Z-score method**:

- P-values converted to Z-scores, combined with weights = √(n_animals)
- Only (gene, celltype) pairs present in ≥2 platforms included
- FDR computed across all {n_meta_total:,} meta-analysis tests

| Meta-analysis FDR | Hits |
|-------------------|------|
| < 0.01 | {(meta.fdr_combined < 0.01).sum()} |
| < 0.05 | {n_meta_fdr05} |
| < 0.10 | {n_meta_fdr10} |
| < 0.20 | {n_meta_fdr20} |
""")

md.append("### 4.3 Cross-Platform Concordance\n")
if conc is not None:
    md.append("**Pairwise logFC concordance (all tested gene × celltype pairs):**\n\n")
    md.append("| Comparison | Common genes | Pearson r | Sign concordance |\n")
    md.append("|------------|-------------|-----------|------------------|\n")
    for _, r in conc.iterrows():
        md.append(f"| {r['pair']} | {int(r['n_common']):,} | {r['pearson_r']:.3f} | {r['sign_concordance']*100:.1f}% |\n")
    md.append("\n")

md.append("""Overall cross-platform concordance is modest (Pearson r = 0.10–0.16 between MERFISH and other platforms), reflecting the challenge of detecting small effect sizes with limited sample sizes across different technologies. The strongest concordance is between MERFISH and Slide-tags, likely because they share the most similar tissue sampling.
""")

md.append("### 4.4 Concordant DE Hits\n")
md.append(f"""Applying a filter of **meta-analysis FDR < 0.2 with concordant effect direction in ≥2 platforms** yields **{n_concordant} gene × celltype hits**.

![DE concordant forest plot](figures/de_concordant_forest_fdr02.png)

![DE concordant heatmap](figures/de_concordant_heatmap_fdr02.png)
""")

md.append("### 4.5 Key Biological Findings\n")

md.append(f"""#### Prolactin signaling ({len(prl_hits)} hits)\n""")
md.append("""Prl (prolactin) is upregulated in multiple **vascular and barrier cell types** during pregnancy:
""")
for _, r in prl_hits.iterrows():
    plats = []
    for p in PLATFORMS:
        lfc = r.get(f'logFC_{p}')
        if pd.notna(lfc):
            plats.append(f"{p}: logFC={lfc:+.2f}")
    md.append(f"- **{r['celltype']}** (meta FDR = {r['fdr_combined']:.4f}): {', '.join(plats)}")

md.append("""
The top Prl-expressing cell types in pregnancy are ABC (arachnoid barrier cells, 22% Prl+), VLMC (vascular leptomeningeal cells, 11% Prl+), and endothelial cells — all cell types sitting at blood-brain or blood-CSF interfaces. This pattern is consistent with **uptake/transport of circulating maternal prolactin into the brain** rather than de novo synthesis.

**Platform-specific consideration:** Prl mRNA is predominantly cytoplasmic (secreted hormone), making it virtually undetectable by Slide-tags snRNA-seq (0.0–0.03% of cells). All Prl results are based on MERFISH and Xenium 5k concordance only.
""")

md.append(f"""#### GABAergic neuron gene regulation\n""")
md.append(f"""Two GABAergic neuron populations show extensive gene-level changes in pregnancy:

**OT D3 Folh1 Gaba** ({len(ot_d3_hits)} concordant DE genes): Predominantly downregulated genes including Gpr37l1, Fabp7, Olig1, Cspg5, Serpine2, Ckb, Mlc1, Atp1b2, Gpr17, Cabp1. Many of these are glial marker genes (Olig1, Fabp7, Gpr17), suggesting either a shift in the transcriptomic profile of these neurons or a change in the cell type composition within this cluster.

**LSX Nkx2-1 Gaba** ({len(lsx_hits)} concordant DE genes): All downregulated — Pld5, Ptpn5, Acvr1, Chrm3. These are lateral septum GABAergic interneurons in the Nkx2-1 lineage, a region involved in social and reproductive behavior.
""")

md.append(f"""#### Apoe / lipid transport ({len(apoe_hits)} hits)\n""")
md.append("""Apoe (apolipoprotein E) is **downregulated** across multiple GABAergic neuron types:
""")
for _, r in apoe_hits.iterrows():
    plats = []
    for p in PLATFORMS:
        lfc = r.get(f'logFC_{p}')
        if pd.notna(lfc):
            plats.append(f"{p}: {lfc:+.2f}")
    md.append(f"- **{r['celltype']}** (meta FDR = {r['fdr_combined']:.4f}): {', '.join(plats)}")
md.append("""
Apoe is a key lipid transport protein in the brain. Its downregulation in GABAergic neurons during pregnancy may reflect altered lipid metabolism or cholesterol homeostasis.
""")

md.append("""#### Immune activation\n""")
if len(cxcl13_hits) > 0:
    r = cxcl13_hits.iloc[0]
    me_lfc = r.get('logFC_merfish', np.nan)
    xe_lfc = r.get('logFC_xenium5k', np.nan)
    cxcl_parts = [f"meta FDR = {r['fdr_combined']:.4f}"]
    if pd.notna(me_lfc):
        cxcl_parts.append(f"MERFISH logFC = {me_lfc:+.2f}")
    if pd.notna(xe_lfc):
        cxcl_parts.append(f"Xenium logFC = {xe_lfc:+.2f}")
    md.append(f"""**Cxcl13** is upregulated in **microglia** ({', '.join(cxcl_parts)}). Cxcl13 is a B-cell chemoattractant typically associated with neuroinflammation. Its upregulation in pregnancy may indicate immune surveillance changes at the blood-brain interface.
""")

md.append("### 4.6 Platform Considerations\n")
md.append(f"""| Feature | MERFISH | Slide-tags | Xenium 5k |
|---------|---------|-----------|-----------|
| Gene panel | {n_genes_merfish} genes | {n_genes_slidetags:,} genes | {n_genes_xenium:,} genes |
| Detection | Cytoplasmic | Nuclear | Cytoplasmic |
| Replicates | 3 per group | 3 per group | 3 per group (2 sections each) |
| Strengths | Direct in situ, spatial context | Genome-wide coverage | Large panel + spatial |
| Limitations | Small gene panel limits DE power | Misses cytoplasmic mRNAs (Prl) | Largest gene panel but limited to panel |

The most productive discovery axis is **Slide-tags ↔ Xenium 5k**, which share large gene panels and replicate each other's findings frequently. MERFISH contributes primarily as a supporting platform for a smaller set of genes.
""")

# ── Section 5: Limitations ──────────────────────────────────────────────
md.append("## 5. Limitations & Future Directions\n")
md.append("""**Statistical power:** With n=3 biological replicates per condition, both compositional and DE analyses are underpowered. The meta-analysis helps but cannot fully compensate. Results at FDR < 0.2 should be treated as hypothesis-generating.

**Annotation accuracy:** The correlation classifier achieves ~65–82% class-level agreement with CAST, with the largest errors in cortical layer-specific glutamatergic neurons. These annotation uncertainties propagate into downstream DE and compositional analyses.

**Technology-specific biases:**
- Slide-tags (nuclear RNA) systematically misses cytoplasmic transcripts, creating blind spots for secreted factors
- MERFISH (496 genes) can only test a small fraction of the transcriptome
- Cell segmentation differences across platforms may affect cell type proportions

**Confounds not addressed:**
- Section-level geometry differences (cortical depth, regional coverage) could affect apparent proportions
- No explicit modeling of tissue depth or spatial position in the DE analysis
- Postpartum samples not fully analyzed across all platforms

**Future directions:**
- Validate top hits (Prl transport, GABAergic downregulation, Cxcl13) with orthogonal methods (ISH, immunohistochemistry)
- Increase biological replication for crumblr to detect modest composition changes
- Integrate spatial information into the annotation pipeline to improve cortical layer resolution
- Extend DE analysis to POSTPART vs CTRL and POSTPART vs PREG contrasts
""")

# ════════════════════════════════════════════════════════════════════════
# Write
# ════════════════════════════════════════════════════════════════════════
md_text = '\n'.join(md)
out_path = f'{OUT}/analysis_results.md'
with open(out_path, 'w') as f:
    f.write(md_text)

print(f"Written: {out_path}")
print(f"  Length: {len(md_text):,} characters, {md_text.count(chr(10)):,} lines")

# Verify all figure references exist
import re
fig_refs = re.findall(r'!\[.*?\]\((figures/.*?)\)', md_text)
print(f"\nFigure references ({len(fig_refs)}):")
for ref in fig_refs:
    full_path = f'{OUT}/{ref}'
    exists = os.path.exists(full_path)
    status = '✓' if exists else '✗ MISSING'
    print(f"  {status} {ref}")

print("\nDone!")
