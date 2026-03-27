#!/usr/bin/env python3
"""
Figures for robust cross-platform DE concordance (PREG vs CTRL).

Filter: FDR < 0.1 in at least one platform AND nominal p < 0.05 in at least
one OTHER platform, with concordant direction across all available platforms.

This is more robust to gene panel differences than requiring nominal significance
in >=2 platforms, because it anchors on one well-powered platform-level result.

Produces:
  1. Forest plot with per-platform logFC + 95% CI, organized by biology
  2. Heatmap of logFC + FDR tiers across platforms
  3. Cell-type-centric summary (which cell types have the most concordant hits)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyBboxPatch
import matplotlib.gridspec as gridspec

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = 'output/de/figures_robust'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────
meta = pd.read_csv('output/de/de_meta_PREG_vs_CTRL.csv')

plat_data = {}
for name, fname in [('merfish', 'de_merfish_PREG_vs_CTRL.csv'),
                     ('slidetags', 'de_slidetags_PREG_vs_CTRL.csv'),
                     ('xenium5k', 'de_xenium5k_PREG_vs_CTRL.csv')]:
    path = f'output/de/{fname}'
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['key'] = df['gene'] + '|' + df['celltype']
        plat_data[name] = df.set_index('key')

PLATFORMS = ['merfish', 'slidetags', 'xenium5k']
PLAT_LABELS = {'merfish': 'MERFISH', 'slidetags': 'Slide-tags', 'xenium5k': 'Xenium 5k'}
PLAT_COLORS = {'merfish': '#E63946', 'slidetags': '#457B9D', 'xenium5k': '#2A9D8F'}
PLAT_OFFSETS = {'merfish': -0.22, 'slidetags': 0.0, 'xenium5k': 0.22}

# ── Filter ─────────────────────────────────────────────────────────────
def apply_robust_filter(row):
    fdrs = {p: row[f'fdr_{p}'] for p in PLATFORMS if pd.notna(row.get(f'fdr_{p}'))}
    pvals = {p: row[f'pval_{p}'] for p in PLATFORMS if pd.notna(row.get(f'pval_{p}'))}
    logfcs = {p: row[f'logFC_{p}'] for p in PLATFORMS if pd.notna(row.get(f'logFC_{p}'))}
    if len(fdrs) < 2:
        return False, '', ''
    signs = [np.sign(v) for v in logfcs.values()]
    if len(set(signs)) > 1:
        return False, '', ''
    if not any(f < 0.1 for f in fdrs.values()):
        return False, '', ''
    anchor = min(fdrs, key=fdrs.get)
    others = {p: pvals[p] for p in pvals if p != anchor}
    if not any(p < 0.05 for p in others.values()):
        return False, '', ''
    replicate = min(others, key=others.get)
    return True, anchor, replicate

results = meta.apply(apply_robust_filter, axis=1, result_type='expand')
meta['passes'] = results[0]
meta['anchor_plat'] = results[1]
meta['replicate_plat'] = results[2]

hits = meta[meta.passes].copy()
hits = hits.sort_values('p_combined').reset_index(drop=True)
print(f"Robust concordant hits: {len(hits)}")

# ── Helpers ────────────────────────────────────────────────────────────
def short_ct(ct):
    parts = ct.split(' ', 1)
    return parts[1][:40] if len(parts) == 2 else ct[:40]

def get_se(gene, celltype, platform):
    key = f'{gene}|{celltype}'
    if platform not in plat_data or key not in plat_data[platform].index:
        return np.nan
    row = plat_data[platform].loc[key]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    f_val = row.get('F', np.nan)
    if pd.notna(f_val) and f_val > 0:
        return abs(row['logFC']) / np.sqrt(f_val)
    return np.nan

def bio_group(gene, ct):
    ct_l = ct.lower()
    if gene == 'Prl':
        return ('A', 'Prolactin signaling')
    if gene == 'Apoe':
        return ('B', 'Apoe / lipid transport')
    if gene == 'Cxcl13':
        return ('C', 'Immune / chemokine')
    if '060 ot d3 folh1' in ct_l:
        return ('D', 'OT D3 Folh1 Gaba cluster')
    if '069 lsx nkx2-1' in ct_l:
        return ('E', 'LSX Nkx2-1 Gaba cluster')
    if any(k in ct_l for k in ['glut']):
        return ('F', 'Other glutamatergic')
    if any(k in ct_l for k in ['gaba']):
        return ('G', 'Other GABAergic')
    return ('H', 'Non-neuronal')

hits[['bio_key', 'bio_label']] = hits.apply(
    lambda r: pd.Series(bio_group(r['gene'], r['celltype'])), axis=1)
hits = hits.sort_values(['bio_key', 'mean_logFC'], ascending=[True, False]).reset_index(drop=True)
hits['label'] = hits['gene'] + ' — ' + hits['celltype'].apply(short_ct)

n = len(hits)
print(f"Biological groups:")
for k, g in hits.groupby('bio_label', sort=False):
    print(f"  {k}: {len(g)} hits")

# ════════════════════════════════════════════════════════════════════════
# FIGURE 1: Forest plot
# ════════════════════════════════════════════════════════════════════════
print("\nFigure 1: Forest plot...")

fig_h = max(8, n * 0.48 + 3)
fig, ax = plt.subplots(figsize=(14, fig_h))

# Category shading
cat_colors_bg = {
    'A': '#FFF3E0', 'B': '#E8F5E9', 'C': '#FCE4EC',
    'D': '#EDE7F6', 'E': '#E3F2FD', 'F': '#FFF8E1',
    'G': '#E0F7FA', 'H': '#F3E5F5',
}
prev_key = None
band_start = 0
for i, (_, row) in enumerate(hits.iterrows()):
    if row['bio_key'] != prev_key:
        if prev_key is not None:
            rect = FancyBboxPatch(
                (-9, band_start - 0.5), 18, i - band_start,
                boxstyle="round,pad=0.05",
                facecolor=cat_colors_bg.get(prev_key, '#fafafa'),
                edgecolor='none', alpha=0.35, zorder=0)
            ax.add_patch(rect)
            mid_y = (band_start + i) / 2
            label_text = hits.iloc[band_start]['bio_label']
            ax.text(-8.5, mid_y, label_text, fontsize=9, fontstyle='italic',
                    color='#555', va='center', ha='left', zorder=1)
        band_start = i
        prev_key = row['bio_key']
# Last band
rect = FancyBboxPatch(
    (-9, band_start - 0.5), 18, n - band_start,
    boxstyle="round,pad=0.05",
    facecolor=cat_colors_bg.get(prev_key, '#fafafa'),
    edgecolor='none', alpha=0.35, zorder=0)
ax.add_patch(rect)
mid_y = (band_start + n) / 2
ax.text(-8.5, mid_y, hits.iloc[band_start]['bio_label'],
        fontsize=9, fontstyle='italic', color='#555', va='center', ha='left', zorder=1)

# Plot points
for i, (_, row) in enumerate(hits.iterrows()):
    for plat in PLATFORMS:
        lfc = row.get(f'logFC_{plat}')
        pval = row.get(f'pval_{plat}')
        fdr = row.get(f'fdr_{plat}')
        if pd.isna(lfc):
            continue

        y = i + PLAT_OFFSETS[plat]
        color = PLAT_COLORS[plat]

        # Marker: filled if anchor/replicate, open if supporting
        is_anchor = plat == row['anchor_plat']
        is_replicate = plat == row['replicate_plat']

        if pd.notna(fdr) and fdr < 0.1:
            marker = 's'  # square = FDR<0.1
            ms = 9
        elif pd.notna(pval) and pval < 0.01:
            marker = 'o'
            ms = 8
        elif pd.notna(pval) and pval < 0.05:
            marker = 'o'
            ms = 6
        else:
            marker = 'o'
            ms = 4

        edge = 'black' if is_anchor else 'none'
        ax.plot(lfc, y, marker, color=color, markersize=ms, zorder=5,
                markeredgecolor=edge, markeredgewidth=1.2 if is_anchor else 0)

        # 95% CI
        se = get_se(row['gene'], row['celltype'], plat)
        if pd.notna(se) and se > 0:
            ci = 1.96 * se
            ax.plot([lfc - ci, lfc + ci], [y, y], '-', color=color,
                    linewidth=1.5, alpha=0.5, zorder=4)

    # Meta diamond
    ax.plot(row['mean_logFC'], i, 'D', color='black', markersize=5,
            zorder=6, alpha=0.7)

ax.axvline(0, color='grey', linewidth=0.8, linestyle='--', zorder=2)
ax.set_yticks(range(n))
ax.set_yticklabels(hits['label'], fontsize=10)
ax.set_xlabel('log₂FC (PREG vs CTRL)', fontsize=14)
ax.set_title(
    'Robust cross-platform concordant DE (PREG vs CTRL)\n'
    'FDR < 0.1 in one platform + p < 0.05 in another + same direction\n'
    'Black-edged square = anchor platform (FDR < 0.1)',
    fontsize=14, fontweight='bold')
ax.invert_yaxis()
ax.set_xlim(-9, 7)

# Right axis: anchor → replicate annotation
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(range(n))
annot = []
for _, row in hits.iterrows():
    a = PLAT_LABELS.get(row['anchor_plat'], '?')[:5]
    r = PLAT_LABELS.get(row['replicate_plat'], '?')[:5]
    a_fdr = row.get(f'fdr_{row["anchor_plat"]}', np.nan)
    r_p = row.get(f'pval_{row["replicate_plat"]}', np.nan)
    annot.append(f'{a} FDR={a_fdr:.3f} → {r} p={r_p:.1e}')
ax2.set_yticklabels(annot, fontsize=8, color='#666', family='monospace')
ax2.tick_params(axis='y', length=0)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=PLAT_COLORS['merfish'],
           markersize=8, label='MERFISH (n=3/group)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=PLAT_COLORS['slidetags'],
           markersize=8, label='Slide-tags (n=3/group)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=PLAT_COLORS['xenium5k'],
           markersize=8, label='Xenium 5k (n=3/group)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='grey',
           markeredgecolor='black', markeredgewidth=1.2, markersize=9,
           label='Anchor: FDR < 0.1'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='black',
           markersize=6, label='Meta-analysis mean'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
          framealpha=0.95, edgecolor='#ccc')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/robust_concordant_forest.png', dpi=250,
            bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved {OUT_DIR}/robust_concordant_forest.png")


# ════════════════════════════════════════════════════════════════════════
# FIGURE 2: Heatmap with FDR tier overlay
# ════════════════════════════════════════════════════════════════════════
print("Figure 2: Heatmap...")

fig, axes = plt.subplots(1, 2, figsize=(12, fig_h * 0.85),
                          gridspec_kw={'width_ratios': [3, 1.2], 'wspace': 0.05})
ax_heat, ax_bar = axes

# logFC heatmap
lfc_mat = np.full((n, 3), np.nan)
for i, (_, row) in enumerate(hits.iterrows()):
    for j, plat in enumerate(PLATFORMS):
        lfc_mat[i, j] = row.get(f'logFC_{plat}', np.nan)

vmax = min(np.nanmax(np.abs(lfc_mat)), 5)
im = ax_heat.imshow(lfc_mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')

# Annotate with FDR tiers
for i, (_, row) in enumerate(hits.iterrows()):
    for j, plat in enumerate(PLATFORMS):
        lfc = row.get(f'logFC_{plat}')
        fdr = row.get(f'fdr_{plat}')
        pval = row.get(f'pval_{plat}')
        if pd.isna(lfc):
            ax_heat.text(j, i, '—', ha='center', va='center', fontsize=8, color='#999')
            continue

        # FDR tier label
        if pd.notna(fdr) and fdr < 0.05:
            txt = f'{lfc:+.1f}\nFDR<.05'
            fw = 'bold'
            fs = 8
        elif pd.notna(fdr) and fdr < 0.1:
            txt = f'{lfc:+.1f}\nFDR<.10'
            fw = 'bold'
            fs = 8
        elif pd.notna(fdr) and fdr < 0.2:
            txt = f'{lfc:+.1f}\nFDR<.20'
            fw = 'normal'
            fs = 7.5
        elif pd.notna(pval) and pval < 0.05:
            txt = f'{lfc:+.1f}\np<.05'
            fw = 'normal'
            fs = 7.5
        else:
            txt = f'{lfc:+.1f}'
            fw = 'normal'
            fs = 7

        # Text color for readability
        tc = 'white' if abs(lfc) > vmax * 0.55 else '#222'
        ax_heat.text(j, i, txt, ha='center', va='center', fontsize=fs,
                     fontweight=fw, color=tc, linespacing=1.1)

ax_heat.set_xticks(range(3))
ax_heat.set_xticklabels(['MERFISH', 'Slide-tags', 'Xenium 5k'],
                         fontsize=12, fontweight='bold')
ax_heat.set_yticks(range(n))
ax_heat.set_yticklabels(hits['label'], fontsize=9.5)

cbar = plt.colorbar(im, ax=ax_heat, shrink=0.4, pad=0.02, location='top')
cbar.set_label('log₂FC', fontsize=10)

# Meta-analysis bar
ax_bar.barh(range(n), hits['mean_logFC'],
            color=['#E63946' if lfc > 0 else '#457B9D' for lfc in hits['mean_logFC']],
            height=0.7, alpha=0.8, edgecolor='white', linewidth=0.5)
ax_bar.axvline(0, color='grey', linewidth=0.8)
ax_bar.set_yticks([])
ax_bar.set_xlabel('Meta logFC', fontsize=10)
ax_bar.invert_yaxis()

# FDR annotation
for i, (_, row) in enumerate(hits.iterrows()):
    fdr = row['fdr_combined']
    star = '***' if fdr < 0.01 else '**' if fdr < 0.05 else '*' if fdr < 0.1 else ''
    lfc = row['mean_logFC']
    offset = 0.15 if lfc > 0 else -0.15
    ha = 'left' if lfc > 0 else 'right'
    ax_bar.text(lfc + offset, i, star, fontsize=9, va='center', ha=ha,
                fontweight='bold', color='#333')

ax_bar.set_title('Meta\n* FDR<.10  ** <.05  *** <.01', fontsize=9, color='#555')

fig.suptitle(
    'Per-platform logFC & FDR for robust concordant DE hits\n'
    'Anchor: FDR < 0.1 in one platform, replicated at p < 0.05 in another',
    fontsize=14, fontweight='bold', y=1.01)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/robust_concordant_heatmap.png', dpi=250,
            bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved {OUT_DIR}/robust_concordant_heatmap.png")


# ════════════════════════════════════════════════════════════════════════
# FIGURE 3: Cell type summary — which cell types have the most hits
# ════════════════════════════════════════════════════════════════════════
print("Figure 3: Cell-type summary...")

ct_counts = hits.groupby('celltype').agg(
    n_genes=('gene', 'count'),
    n_up=('mean_logFC', lambda x: (x > 0).sum()),
    n_down=('mean_logFC', lambda x: (x < 0).sum()),
    genes_up=('gene', lambda x: ', '.join(
        hits.loc[x.index[hits.loc[x.index, 'mean_logFC'] > 0], 'gene'].tolist())),
    genes_down=('gene', lambda x: ', '.join(
        hits.loc[x.index[hits.loc[x.index, 'mean_logFC'] < 0], 'gene'].tolist())),
).reset_index()
ct_counts = ct_counts[ct_counts.n_genes >= 1].sort_values('n_genes', ascending=True)
ct_counts['ct_short'] = ct_counts['celltype'].apply(short_ct)

# Only show cell types with >=2 hits for this figure
ct_multi = ct_counts[ct_counts.n_genes >= 2].copy()

if len(ct_multi) > 0:
    fig, ax = plt.subplots(figsize=(12, max(4, len(ct_multi) * 0.8 + 2)))

    y = range(len(ct_multi))
    ax.barh(y, ct_multi['n_up'], color='#E63946', alpha=0.8, label='Upregulated',
            height=0.6, edgecolor='white')
    ax.barh(y, -ct_multi['n_down'], color='#457B9D', alpha=0.8, label='Downregulated',
            height=0.6, edgecolor='white')

    # Gene name annotations
    for i, (_, row) in enumerate(ct_multi.iterrows()):
        if row['genes_up']:
            ax.text(row['n_up'] + 0.15, i, row['genes_up'],
                    fontsize=9, va='center', color='#C62828', fontstyle='italic')
        if row['genes_down']:
            ax.text(-row['n_down'] - 0.15, i, row['genes_down'],
                    fontsize=9, va='center', ha='right', color='#1565C0', fontstyle='italic')

    ax.axvline(0, color='black', linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(ct_multi['ct_short'], fontsize=12)
    ax.set_xlabel('Number of concordant DE genes', fontsize=13)
    ax.set_title('Cell types with multiple robust concordant DE genes\n'
                 'FDR < 0.1 in one platform + p < 0.05 in another + same direction',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/robust_concordant_by_celltype.png', dpi=250,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved {OUT_DIR}/robust_concordant_by_celltype.png")


# ════════════════════════════════════════════════════════════════════════
# FIGURE 4: Anchor platform breakdown — which platforms drive discoveries
# ════════════════════════════════════════════════════════════════════════
print("Figure 4: Anchor platform breakdown...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: anchor platform counts
anchor_counts = hits['anchor_plat'].value_counts()
axes[0].bar(range(len(anchor_counts)),
            [anchor_counts.get(p, 0) for p in PLATFORMS],
            color=[PLAT_COLORS[p] for p in PLATFORMS],
            width=0.6, edgecolor='white')
axes[0].set_xticks(range(len(PLATFORMS)))
axes[0].set_xticklabels([PLAT_LABELS[p] for p in PLATFORMS], fontsize=12)
axes[0].set_ylabel('Number of anchor discoveries', fontsize=12)
axes[0].set_title('Which platform drives discoveries?\n(FDR < 0.1 anchor)',
                   fontsize=13, fontweight='bold')

for i, p in enumerate(PLATFORMS):
    cnt = anchor_counts.get(p, 0)
    axes[0].text(i, cnt + 0.3, str(cnt), ha='center', fontsize=14, fontweight='bold')

# Panel B: replication pair matrix
pair_counts = hits.groupby(['anchor_plat', 'replicate_plat']).size().reset_index(name='count')
pair_mat = np.zeros((3, 3))
for _, r in pair_counts.iterrows():
    ai = PLATFORMS.index(r['anchor_plat'])
    ri = PLATFORMS.index(r['replicate_plat'])
    pair_mat[ai, ri] = r['count']

im2 = axes[1].imshow(pair_mat, cmap='YlOrRd', vmin=0)
for i in range(3):
    for j in range(3):
        v = int(pair_mat[i, j])
        if v > 0:
            axes[1].text(j, i, str(v), ha='center', va='center',
                         fontsize=16, fontweight='bold',
                         color='white' if v > 5 else 'black')
        elif i == j:
            axes[1].text(j, i, '—', ha='center', va='center', fontsize=12, color='#999')

axes[1].set_xticks(range(3))
axes[1].set_xticklabels([PLAT_LABELS[p] for p in PLATFORMS], fontsize=11)
axes[1].set_yticks(range(3))
axes[1].set_yticklabels([PLAT_LABELS[p] for p in PLATFORMS], fontsize=11)
axes[1].set_xlabel('Replicate (p < 0.05)', fontsize=11)
axes[1].set_ylabel('Anchor (FDR < 0.1)', fontsize=11)
axes[1].set_title('Anchor → Replicate pairs', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/robust_concordant_platform_breakdown.png', dpi=250,
            bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved {OUT_DIR}/robust_concordant_platform_breakdown.png")


# ════════════════════════════════════════════════════════════════════════
# Save table
# ════════════════════════════════════════════════════════════════════════
out_cols = ['gene', 'celltype', 'bio_label', 'mean_logFC', 'fdr_combined',
            'anchor_plat', 'replicate_plat', 'n_platforms',
            'logFC_merfish', 'pval_merfish', 'fdr_merfish',
            'logFC_slidetags', 'pval_slidetags', 'fdr_slidetags',
            'logFC_xenium5k', 'pval_xenium5k', 'fdr_xenium5k']
hits[out_cols].to_csv(f'{OUT_DIR}/robust_concordant_hits.csv', index=False)
print(f"\nSaved: {OUT_DIR}/robust_concordant_hits.csv ({len(hits)} rows)")

print("\nDone!")
