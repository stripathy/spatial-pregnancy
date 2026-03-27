#!/usr/bin/env python3
"""
Figures for concordant cross-platform DE hits (PREG vs CTRL).

Filters: concordant direction across all available platforms,
         nominal p < 0.05 in >= 2 platforms, meta FDR < 0.20.

Produces:
  1. Forest plot of top hits (per-platform logFC + CI)
  2. Heatmap of logFC organized by cell type class + biology
  3. Cross-platform scatter for top cell types
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = 'output/de/figures'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────
meta = pd.read_csv('output/de/de_meta_PREG_vs_CTRL.csv')

# Per-platform data for SEs
plat_data = {}
for name, fname in [('merfish', 'de_merfish_PREG_vs_CTRL.csv'),
                     ('slidetags', 'de_slidetags_PREG_vs_CTRL.csv'),
                     ('xenium5k', 'de_xenium5k_PREG_vs_CTRL.csv')]:
    path = f'output/de/{fname}'
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['key'] = df['gene'] + '|' + df['celltype']
        plat_data[name] = df.set_index('key')

# ── Filter concordant hits ─────────────────────────────────────────────
def check_concordant(row):
    platforms = ['merfish', 'slidetags', 'xenium5k']
    pvals = {p: row[f'pval_{p}'] for p in platforms if pd.notna(row.get(f'pval_{p}'))}
    logfcs = {p: row[f'logFC_{p}'] for p in platforms if pd.notna(row.get(f'logFC_{p}'))}
    if len(pvals) < 2:
        return False
    signs = [np.sign(v) for v in logfcs.values()]
    if len(set(signs)) > 1:
        return False
    n_sig = sum(1 for p in pvals.values() if p < 0.05)
    return n_sig >= 2

meta['passes'] = meta.apply(check_concordant, axis=1)
hits = meta[meta.passes & (meta.fdr_combined < 0.20)].copy()
hits = hits.sort_values('p_combined').reset_index(drop=True)

print(f"Concordant hits (meta FDR<0.20): {len(hits)}")

# ── Parse cell type class from taxonomy number ─────────────────────────
def get_class(ct):
    """Extract broad class from ABCA taxonomy name."""
    ct_lower = ct.lower()
    if any(k in ct_lower for k in ['glut']):
        return 'Glutamatergic'
    elif any(k in ct_lower for k in ['gaba']):
        return 'GABAergic'
    elif any(k in ct_lower for k in ['astro', 'astroependymal']):
        return 'Astrocyte'
    elif any(k in ct_lower for k in ['oligo']):
        return 'Oligodendrocyte'
    elif any(k in ct_lower for k in ['opc']):
        return 'OPC'
    elif any(k in ct_lower for k in ['micro']):
        return 'Microglia'
    elif any(k in ct_lower for k in ['endo']):
        return 'Endothelial'
    elif any(k in ct_lower for k in ['vlmc']):
        return 'VLMC'
    elif any(k in ct_lower for k in ['peri']):
        return 'Pericyte'
    elif any(k in ct_lower for k in ['abc']):
        return 'ABC'
    elif any(k in ct_lower for k in ['chor']):
        return 'Choroid plexus'
    else:
        return 'Other NN'

hits['broad_class'] = hits['celltype'].apply(get_class)

# Biological category for grouping
def bio_category(row):
    gene = row['gene']
    cls = row['broad_class']
    if gene == 'Prl':
        return '1_Prolactin_signaling'
    elif gene == 'Apoe':
        return '2_Apoe_lipid_metabolism'
    elif gene in ('Cxcl13',):
        return '3_Immune_activation'
    elif cls in ('Glutamatergic', 'GABAergic'):
        return '4_Neuronal_gene_regulation'
    else:
        return '5_Non_neuronal_changes'

hits['bio_cat'] = hits.apply(bio_category, axis=1)
hits = hits.sort_values(['bio_cat', 'mean_logFC'], ascending=[True, False]).reset_index(drop=True)

# ── Shorten cell type names ────────────────────────────────────────────
def short_ct(ct):
    parts = ct.split(' ', 1)
    if len(parts) == 2:
        return parts[1][:35]
    return ct[:35]

hits['ct_short'] = hits['celltype'].apply(short_ct)
hits['label'] = hits['gene'] + ' — ' + hits['ct_short']

# ── Get SE from per-platform data ──────────────────────────────────────
def get_se(gene, celltype, platform):
    """Get standard error from per-platform edgeR output."""
    key = f'{gene}|{celltype}'
    if platform not in plat_data:
        return np.nan
    df = plat_data[platform]
    if key not in df.index:
        return np.nan
    row = df.loc[key]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    logfc = row['logFC']
    # F statistic: F = (logFC/SE)^2 → SE = |logFC|/sqrt(F)
    f_val = row.get('F', np.nan)
    if pd.notna(f_val) and f_val > 0:
        return abs(logfc) / np.sqrt(f_val)
    return np.nan


# ════════════════════════════════════════════════════════════════════════
# FIGURE 1: Forest plot
# ════════════════════════════════════════════════════════════════════════
print("Creating forest plot...")

n_hits = len(hits)
fig_height = max(10, n_hits * 0.45 + 2)
fig, ax = plt.subplots(figsize=(14, fig_height))

platform_colors = {'merfish': '#E63946', 'slidetags': '#457B9D', 'xenium5k': '#2A9D8F'}
platform_labels = {'merfish': 'MERFISH', 'slidetags': 'Slide-tags', 'xenium5k': 'Xenium 5k'}
platform_offsets = {'merfish': -0.2, 'slidetags': 0.0, 'xenium5k': 0.2}

# Bio category backgrounds
cat_labels = {
    '1_Prolactin_signaling': 'Prolactin signaling',
    '2_Apoe_lipid_metabolism': 'Apoe / lipid metabolism',
    '3_Immune_activation': 'Immune activation',
    '4_Neuronal_gene_regulation': 'Neuronal gene regulation',
    '5_Non_neuronal_changes': 'Non-neuronal changes',
}
cat_colors = {
    '1_Prolactin_signaling': '#FFF3E0',
    '2_Apoe_lipid_metabolism': '#E8F5E9',
    '3_Immune_activation': '#FCE4EC',
    '4_Neuronal_gene_regulation': '#E3F2FD',
    '5_Non_neuronal_changes': '#F3E5F5',
}

# Draw category bands
prev_cat = None
cat_start = 0
for i, row in hits.iterrows():
    if row['bio_cat'] != prev_cat:
        if prev_cat is not None:
            rect = FancyBboxPatch((-8, cat_start - 0.5), 16, i - cat_start,
                                  boxstyle="round,pad=0.05",
                                  facecolor=cat_colors.get(prev_cat, '#f5f5f5'),
                                  edgecolor='none', alpha=0.3, zorder=0)
            ax.add_patch(rect)
            # Category label
            mid = (cat_start + i) / 2
            ax.text(-7.5, mid, cat_labels.get(prev_cat, ''),
                    fontsize=9, fontstyle='italic', color='#555', va='center',
                    ha='left', zorder=1)
        cat_start = i
        prev_cat = row['bio_cat']
# Last category
rect = FancyBboxPatch((-8, cat_start - 0.5), 16, n_hits - cat_start,
                      boxstyle="round,pad=0.05",
                      facecolor=cat_colors.get(prev_cat, '#f5f5f5'),
                      edgecolor='none', alpha=0.3, zorder=0)
ax.add_patch(rect)
mid = (cat_start + n_hits) / 2
ax.text(-7.5, mid, cat_labels.get(prev_cat, ''),
        fontsize=9, fontstyle='italic', color='#555', va='center', ha='left', zorder=1)

# Plot per-platform effect sizes
for i, (_, row) in enumerate(hits.iterrows()):
    for plat in ['merfish', 'slidetags', 'xenium5k']:
        lfc = row.get(f'logFC_{plat}')
        pval = row.get(f'pval_{plat}')
        if pd.isna(lfc):
            continue
        se = get_se(row['gene'], row['celltype'], plat)
        y = i + platform_offsets[plat]
        color = platform_colors[plat]

        # Marker size based on significance
        if pd.notna(pval) and pval < 0.001:
            ms = 9
        elif pd.notna(pval) and pval < 0.01:
            ms = 7
        elif pd.notna(pval) and pval < 0.05:
            ms = 5
        else:
            ms = 3

        ax.plot(lfc, y, 'o', color=color, markersize=ms, zorder=5)

        # Error bar (95% CI)
        if pd.notna(se) and se > 0:
            ci = 1.96 * se
            ax.plot([lfc - ci, lfc + ci], [y, y], '-', color=color,
                    linewidth=1.5, alpha=0.6, zorder=4)

    # Meta-analysis diamond
    meta_lfc = row['mean_logFC']
    ax.plot(meta_lfc, i, 'D', color='black', markersize=6, zorder=6, alpha=0.8)

ax.axvline(0, color='grey', linewidth=0.8, linestyle='--', zorder=2)
ax.set_yticks(range(n_hits))
ax.set_yticklabels(hits['label'], fontsize=11)
ax.set_xlabel('log₂FC (PREG vs CTRL)', fontsize=14)
ax.set_title('Cross-platform concordant DE hits (PREG vs CTRL)\n'
             'Meta FDR < 0.20 | Concordant direction | Nominal p < 0.05 in ≥2 platforms',
             fontsize=16, fontweight='bold')

# FDR annotation on right
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(range(n_hits))
fdr_labels = [f"FDR={row['fdr_combined']:.3f}" for _, row in hits.iterrows()]
ax2.set_yticklabels(fdr_labels, fontsize=9, color='#666')
ax2.tick_params(axis='y', length=0)

# Legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=platform_colors['merfish'],
           markersize=8, label='MERFISH (n=3/group)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=platform_colors['slidetags'],
           markersize=8, label='Slide-tags (n=3/group)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=platform_colors['xenium5k'],
           markersize=8, label='Xenium 5k (n=3/group)'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='black',
           markersize=7, label='Meta-analysis mean'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='grey',
           markersize=9, label='p < 0.001'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='grey',
           markersize=7, label='p < 0.01'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='grey',
           markersize=5, label='p < 0.05'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10,
          framealpha=0.95, edgecolor='#ccc')

ax.invert_yaxis()
ax.set_xlim(-8, 7)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/de_concordant_forest.png', dpi=250, bbox_inches='tight',
            facecolor='white')
plt.close()
print(f"  Saved {OUT_DIR}/de_concordant_forest.png")


# ════════════════════════════════════════════════════════════════════════
# FIGURE 2: Heatmap of logFC across platforms
# ════════════════════════════════════════════════════════════════════════
print("Creating heatmap...")

fig, ax = plt.subplots(figsize=(8, fig_height * 0.8))

platforms = ['merfish', 'slidetags', 'xenium5k']
lfc_matrix = np.full((n_hits, 3), np.nan)
for i, (_, row) in enumerate(hits.iterrows()):
    for j, plat in enumerate(platforms):
        lfc_matrix[i, j] = row.get(f'logFC_{plat}', np.nan)

# Color scale
vmax = np.nanmax(np.abs(lfc_matrix))
vmax = min(vmax, 5)

im = ax.imshow(lfc_matrix, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')

# Significance markers
for i, (_, row) in enumerate(hits.iterrows()):
    for j, plat in enumerate(platforms):
        pval = row.get(f'pval_{plat}')
        if pd.isna(pval) or pd.isna(row.get(f'logFC_{plat}')):
            ax.text(j, i, '—', ha='center', va='center', fontsize=8, color='#999')
        elif pval < 0.001:
            ax.text(j, i, '***', ha='center', va='center', fontsize=9, fontweight='bold')
        elif pval < 0.01:
            ax.text(j, i, '**', ha='center', va='center', fontsize=9, fontweight='bold')
        elif pval < 0.05:
            ax.text(j, i, '*', ha='center', va='center', fontsize=9)
        else:
            lfc_val = row.get(f'logFC_{plat}')
            if pd.notna(lfc_val):
                ax.text(j, i, f'{lfc_val:.1f}', ha='center', va='center',
                        fontsize=7, color='#333')

ax.set_xticks(range(3))
ax.set_xticklabels(['MERFISH', 'Slide-tags', 'Xenium 5k'], fontsize=12, fontweight='bold')
ax.set_yticks(range(n_hits))
ax.set_yticklabels(hits['label'], fontsize=10)
ax.set_title('Per-platform logFC for concordant DE hits\n'
             '* p<0.05  ** p<0.01  *** p<0.001  — not tested',
             fontsize=14, fontweight='bold')

cbar = plt.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
cbar.set_label('log₂FC (PREG vs CTRL)', fontsize=11)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/de_concordant_heatmap.png', dpi=250, bbox_inches='tight',
            facecolor='white')
plt.close()
print(f"  Saved {OUT_DIR}/de_concordant_heatmap.png")


# ════════════════════════════════════════════════════════════════════════
# FIGURE 3: Prl across cell types (pregnancy hormone highlight)
# ════════════════════════════════════════════════════════════════════════
print("Creating Prl highlight figure...")

prl_meta = meta[(meta.gene == 'Prl') & (meta.concordant == True)].copy()
prl_meta = prl_meta.sort_values('p_combined')
# Take top 10 cell types for Prl
prl_top = prl_meta.head(10).copy()
prl_top['ct_short'] = prl_top['celltype'].apply(short_ct)
prl_top = prl_top.sort_values('mean_logFC', ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, 6))

for i, (_, row) in enumerate(prl_top.iterrows()):
    for plat in ['merfish', 'slidetags', 'xenium5k']:
        lfc = row.get(f'logFC_{plat}')
        pval = row.get(f'pval_{plat}')
        if pd.isna(lfc):
            continue
        color = platform_colors[plat]
        ms = 8 if pd.notna(pval) and pval < 0.01 else 5
        ax.plot(lfc, i + platform_offsets[plat], 'o', color=color, markersize=ms, zorder=5)

    ax.plot(row['mean_logFC'], i, 'D', color='black', markersize=7, zorder=6)

ax.axvline(0, color='grey', linewidth=0.8, linestyle='--')
ax.set_yticks(range(len(prl_top)))
ax.set_yticklabels(prl_top['ct_short'], fontsize=12)
ax.set_xlabel('log₂FC (PREG vs CTRL)', fontsize=14)
ax.set_title('Prl (Prolactin) upregulation across cell types in pregnancy\n'
             'Concordant direction across platforms',
             fontsize=15, fontweight='bold')

# FDR on right
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(range(len(prl_top)))
ax2.set_yticklabels([f"FDR={row['fdr_combined']:.3f}" for _, row in prl_top.iterrows()],
                    fontsize=9, color='#666')
ax2.tick_params(axis='y', length=0)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=platform_colors['merfish'],
           markersize=8, label='MERFISH'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=platform_colors['slidetags'],
           markersize=8, label='Slide-tags'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=platform_colors['xenium5k'],
           markersize=8, label='Xenium 5k'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='black',
           markersize=7, label='Meta-analysis mean'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/de_prl_across_celltypes.png', dpi=250, bbox_inches='tight',
            facecolor='white')
plt.close()
print(f"  Saved {OUT_DIR}/de_prl_across_celltypes.png")


# ════════════════════════════════════════════════════════════════════════
# FIGURE 4: OT D3 Folh1 Gaba cluster — many downregulated genes
# ════════════════════════════════════════════════════════════════════════
print("Creating OT D3 Folh1 Gaba cluster figure...")

ot_hits = hits[hits.celltype.str.contains('060 OT D3 Folh1')].copy()
ot_hits = ot_hits.sort_values('mean_logFC').reset_index(drop=True)

if len(ot_hits) > 0:
    fig, ax = plt.subplots(figsize=(10, max(4, len(ot_hits) * 0.55 + 1.5)))

    for i, (_, row) in enumerate(ot_hits.iterrows()):
        for plat in ['merfish', 'slidetags', 'xenium5k']:
            lfc = row.get(f'logFC_{plat}')
            pval = row.get(f'pval_{plat}')
            if pd.isna(lfc):
                continue
            color = platform_colors[plat]
            ms = 8 if pd.notna(pval) and pval < 0.01 else 5
            ax.plot(lfc, i + platform_offsets[plat], 'o', color=color, markersize=ms, zorder=5)

            se = get_se(row['gene'], row['celltype'], plat)
            if pd.notna(se) and se > 0:
                ci = 1.96 * se
                ax.plot([lfc - ci, lfc + ci], [i + platform_offsets[plat]] * 2,
                        '-', color=color, linewidth=1.5, alpha=0.5, zorder=4)

        ax.plot(row['mean_logFC'], i, 'D', color='black', markersize=7, zorder=6)

    ax.axvline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.set_yticks(range(len(ot_hits)))
    ax.set_yticklabels([f"{r['gene']}  (FDR={r['fdr_combined']:.3f})" for _, r in ot_hits.iterrows()],
                       fontsize=12)
    ax.set_xlabel('log₂FC (PREG vs CTRL)', fontsize=14)
    ax.set_title('060 OT D3 Folh1 Gaba: multiple genes downregulated in pregnancy\n'
                 'Concordant across Slide-tags + Xenium 5k',
                 fontsize=14, fontweight='bold')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=platform_colors['merfish'],
               markersize=8, label='MERFISH'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=platform_colors['slidetags'],
               markersize=8, label='Slide-tags'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=platform_colors['xenium5k'],
               markersize=8, label='Xenium 5k'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='black',
               markersize=7, label='Meta mean'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/de_OT_D3_Folh1_Gaba_cluster.png', dpi=250,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved {OUT_DIR}/de_OT_D3_Folh1_Gaba_cluster.png")


# ════════════════════════════════════════════════════════════════════════
# FIGURE 5: Summary barplot — grouped by biological theme
# ════════════════════════════════════════════════════════════════════════
print("Creating summary barplot...")

fig, ax = plt.subplots(figsize=(12, 8))

# Reorder for nice display: upregulated on right, downregulated on left
hits_sorted = hits.sort_values('mean_logFC', ascending=True).reset_index(drop=True)

colors_bar = []
for _, row in hits_sorted.iterrows():
    if row['gene'] == 'Prl':
        colors_bar.append('#FF6B6B')
    elif row['gene'] == 'Apoe':
        colors_bar.append('#51CF66')
    elif row['gene'] == 'Cxcl13':
        colors_bar.append('#FF922B')
    elif row['mean_logFC'] > 0:
        colors_bar.append('#E63946')
    else:
        colors_bar.append('#457B9D')

bars = ax.barh(range(len(hits_sorted)), hits_sorted['mean_logFC'], color=colors_bar,
               edgecolor='white', linewidth=0.5, height=0.7, alpha=0.85)

# Add significance stars
for i, (_, row) in enumerate(hits_sorted.iterrows()):
    fdr = row['fdr_combined']
    lfc = row['mean_logFC']
    star = '***' if fdr < 0.01 else '**' if fdr < 0.05 else '*' if fdr < 0.10 else ''
    offset = 0.1 if lfc > 0 else -0.1
    ha = 'left' if lfc > 0 else 'right'
    ax.text(lfc + offset, i, star, fontsize=10, va='center', ha=ha, fontweight='bold', color='#333')

    # n_platforms annotation
    n_plat = int(row['n_platforms'])
    ax.text(lfc + offset * 3, i, f'({n_plat}p)', fontsize=7, va='center', ha=ha, color='#888')

ax.axvline(0, color='black', linewidth=1)
ax.set_yticks(range(len(hits_sorted)))
ax.set_yticklabels([f"{r['gene']} — {short_ct(r['celltype'])}"
                    for _, r in hits_sorted.iterrows()], fontsize=10)
ax.set_xlabel('Meta-analysis mean log₂FC (PREG vs CTRL)', fontsize=14)
ax.set_title('Concordant cross-platform DE genes in pregnancy\n'
             'Sorted by effect size | * FDR<0.10  ** FDR<0.05  *** FDR<0.01 | (Np) = platforms',
             fontsize=15, fontweight='bold')

# Highlight legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#FF6B6B', label='Prl (Prolactin)'),
    Patch(facecolor='#51CF66', label='Apoe (Lipid metabolism)'),
    Patch(facecolor='#FF922B', label='Cxcl13 (Immune)'),
    Patch(facecolor='#E63946', label='Other upregulated'),
    Patch(facecolor='#457B9D', label='Other downregulated'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10, framealpha=0.95)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/de_concordant_summary_bar.png', dpi=250, bbox_inches='tight',
            facecolor='white')
plt.close()
print(f"  Saved {OUT_DIR}/de_concordant_summary_bar.png")


# ════════════════════════════════════════════════════════════════════════
# Save filtered table
# ════════════════════════════════════════════════════════════════════════
out_cols = ['gene', 'celltype', 'broad_class', 'bio_cat', 'mean_logFC', 'fdr_combined',
            'n_platforms', 'concordant',
            'logFC_merfish', 'pval_merfish', 'fdr_merfish',
            'logFC_slidetags', 'pval_slidetags', 'fdr_slidetags',
            'logFC_xenium5k', 'pval_xenium5k', 'fdr_xenium5k']
hits[out_cols].to_csv(f'{OUT_DIR}/de_concordant_hits_table.csv', index=False)
print(f"\nSaved filtered table: {OUT_DIR}/de_concordant_hits_table.csv ({len(hits)} rows)")

print("\nDone!")
