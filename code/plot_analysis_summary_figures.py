#!/usr/bin/env python3
"""
Generate all figures needed for the analysis summary markdown.

Produces:
  1. Crumblr proportion boxplots (suggestive cell types, per platform)
  2. Xenium 5k validation scatter plots (class + subclass proportions)
  3. DE meta-analysis concordant forest plot (FDR<0.2 + concordant ≥2 platforms)
  4. DE concordant heatmap
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from scipy.stats import pearsonr
import glob

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = 'output/analysis_summary/figures'
os.makedirs(OUT, exist_ok=True)

PLAT_COLORS = {'merfish': '#E63946', 'slidetags': '#457B9D', 'xenium5k': '#2A9D8F'}
PLAT_LABELS = {'merfish': 'MERFISH', 'slidetags': 'Slide-tags', 'xenium5k': 'Xenium 5k'}
COND_COLORS = {'CTRL': '#4A90D9', 'PREG': '#E85D75', 'POSTPART': '#7AC074'}
COND_ORDER = ['CTRL', 'PREG', 'POSTPART']
PLATFORMS = ['merfish', 'slidetags', 'xenium5k']


# ════════════════════════════════════════════════════════════════════════
# FIGURE 1: Crumblr proportion boxplots
# ════════════════════════════════════════════════════════════════════════
def plot_crumblr_boxplots():
    print("Figure 1: Crumblr proportion boxplots...")

    cr = pd.read_csv('output/crumblr/crumblr_results_all.csv')
    preg = cr[cr.contrast == 'PREG_vs_CTRL']

    # Top suggestive cell types (from Xenium subclass level, which had most power)
    target_cts = [
        '066 NDB-SI-ant Prdm12 Gaba',
        '322 Tanycyte NN',
        '060 OT D3 Folh1 Gaba',
        '336 Monocytes NN',
        '321 Astroependymal NN',
        '088 BST Tac2 Gaba',
    ]

    # Load crumblr input data for each platform (hier_subclass level)
    plat_inputs = {}
    for plat in PLATFORMS:
        # Try hier_subclass first, fall back to subclass
        for suffix in ['hier_subclass', 'subclass']:
            path = f'output/crumblr/crumblr_input_{plat}_{suffix}.csv'
            if os.path.exists(path):
                plat_inputs[plat] = pd.read_csv(path)
                plat_inputs[plat]['proportion'] = plat_inputs[plat]['count'] / plat_inputs[plat]['total']
                break

    # Get crumblr p-values for annotation
    def get_pval(plat, ct):
        levels = [f'{plat}_hier_subclass', f'{plat}_subclass']
        for lev in levels:
            row = preg[(preg.level == lev) & (preg.celltype == ct)]
            if len(row) > 0:
                return row.iloc[0]['P.Value']
        return np.nan

    # Multi-panel figure: one row per cell type, one column per platform
    n_cts = len(target_cts)
    fig, axes = plt.subplots(n_cts, 3, figsize=(14, n_cts * 2.5 + 2),
                              squeeze=False)

    for i, ct in enumerate(target_cts):
        for j, plat in enumerate(PLATFORMS):
            ax = axes[i, j]
            if plat not in plat_inputs:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                        ha='center', va='center', fontsize=10, color='#999')
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            df = plat_inputs[plat]
            ct_data = df[df.celltype == ct]

            if len(ct_data) == 0:
                ax.text(0.5, 0.5, 'Not tested', transform=ax.transAxes,
                        ha='center', va='center', fontsize=10, color='#999')
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(PLAT_LABELS[plat], fontsize=14, fontweight='bold')
                continue

            # Available conditions
            conds = [c for c in COND_ORDER if c in ct_data.condition.values]
            positions = list(range(len(conds)))
            bp_data = [ct_data[ct_data.condition == c]['proportion'].values * 100
                       for c in conds]

            bp = ax.boxplot(bp_data, positions=positions, widths=0.5,
                           patch_artist=True, showfliers=True,
                           medianprops={'color': 'black', 'linewidth': 1.5})

            for k, patch in enumerate(bp['boxes']):
                patch.set_facecolor(COND_COLORS[conds[k]])
                patch.set_alpha(0.7)

            # Overlay individual points
            for k, c in enumerate(conds):
                vals = ct_data[ct_data.condition == c]['proportion'].values * 100
                jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
                ax.scatter(k + jitter, vals, color=COND_COLORS[c],
                          edgecolor='black', linewidth=0.5, s=30, zorder=5, alpha=0.9)

            ax.set_xticks(positions)
            ax.set_xticklabels(conds, fontsize=9)

            # p-value annotation
            pval = get_pval(plat, ct)
            if pd.notna(pval):
                ax.text(0.98, 0.95, f'p={pval:.2e}', transform=ax.transAxes,
                        ha='right', va='top', fontsize=8, color='#666',
                        fontstyle='italic')

            if i == 0:
                ax.set_title(PLAT_LABELS[plat], fontsize=14, fontweight='bold')
            if j == 0:
                # Short cell type name
                ct_short = ct.split(' ', 1)[1][:30] if ' ' in ct else ct[:30]
                ax.set_ylabel(ct_short, fontsize=10, fontweight='bold')
            else:
                ax.set_ylabel('')

            ax.tick_params(axis='y', labelsize=9)

    # Shared y-label
    fig.text(0.02, 0.5, 'Proportion (%)', va='center', rotation='vertical',
             fontsize=13)

    fig.suptitle('Cell type proportions for suggestive crumblr hits (PREG vs CTRL)\n'
                 'Each dot = one tissue section/sample',
                 fontsize=14, fontweight='bold', y=1.01)

    plt.tight_layout(rect=[0.04, 0, 1, 0.98])
    plt.savefig(f'{OUT}/crumblr_proportion_boxplots.png', dpi=250,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved {OUT}/crumblr_proportion_boxplots.png")


# ════════════════════════════════════════════════════════════════════════
# FIGURE 2: Xenium 5k validation scatters
# ════════════════════════════════════════════════════════════════════════
def plot_xenium_validation():
    print("Figure 2: Xenium validation scatters...")

    class_props = pd.read_csv('output/xenium5k/class_proportions_all_modalities.csv')
    sub_props = pd.read_csv('output/xenium5k/subclass_proportions_all_modalities.csv')

    # Rename unnamed column
    class_props = class_props.rename(columns={class_props.columns[0]: 'celltype'})
    sub_props = sub_props.rename(columns={sub_props.columns[0]: 'celltype'})

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel A: Class proportions — Xenium vs Reference
    ax = axes[0]
    ref = class_props['Reference'].values
    xen = class_props['Xenium5k_Hier'].values
    mask = (ref > 0) & (xen > 0)
    r, _ = pearsonr(np.log10(ref[mask]), np.log10(xen[mask]))
    ax.scatter(ref * 100, xen * 100, s=60, c='#2A9D8F', edgecolor='black',
              linewidth=0.5, zorder=5)
    # Identity line
    lims = [0.01, max(ref.max(), xen.max()) * 100 * 1.5]
    ax.plot(lims, lims, '--', color='grey', linewidth=1, zorder=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Zeng Reference (%)', fontsize=12)
    ax.set_ylabel('Xenium 5k (%)', fontsize=12)
    ax.set_title(f'Class proportions\nr = {r:.3f} (log₁₀)', fontsize=13, fontweight='bold')
    # Label top types
    for _, row in class_props.nlargest(6, 'Reference').iterrows():
        ax.annotate(row['celltype'][:20], (row['Reference'] * 100, row['Xenium5k_Hier'] * 100),
                   fontsize=7, ha='left', xytext=(5, 3), textcoords='offset points')

    # Panel B: Subclass proportions — Xenium vs Reference
    ax = axes[1]
    ref = sub_props['Reference'].values
    xen = sub_props['Xenium5k'].values
    mask = (ref > 0) & (xen > 0)
    r2, _ = pearsonr(np.log10(ref[mask] + 1e-6), np.log10(xen[mask] + 1e-6))
    ax.scatter(ref[mask] * 100, xen[mask] * 100, s=30, c='#2A9D8F', edgecolor='black',
              linewidth=0.3, alpha=0.7, zorder=5)
    ax.plot([0.001, 100], [0.001, 100], '--', color='grey', linewidth=1, zorder=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Zeng Reference (%)', fontsize=12)
    ax.set_ylabel('Xenium 5k (%)', fontsize=12)
    ax.set_title(f'Subclass proportions (n={mask.sum()} types)\nr = {r2:.3f} (log₁₀)',
                fontsize=13, fontweight='bold')

    # Panel C: All platforms class-level comparison
    ax = axes[2]
    modalities = {
        'Xenium5k_Hier': ('#2A9D8F', 'Xenium 5k'),
        'MERFISH_Hier': ('#E63946', 'MERFISH'),
        'MERFISH_CAST': ('#E6394680', 'MERFISH (CAST)'),
        'SlideTag_Hier': ('#457B9D', 'Slide-tags'),
        'SlideTag_CAST': ('#457B9D80', 'Slide-tags (CAST)'),
    }
    ref_vals = class_props['Reference'].values
    for col, (color, label) in modalities.items():
        if col in class_props.columns:
            vals = class_props[col].values
            m = (ref_vals > 0) & (vals > 0)
            r_val, _ = pearsonr(np.log10(ref_vals[m]), np.log10(vals[m]))
            ax.scatter(ref_vals * 100, vals * 100, s=40, c=color, edgecolor='black',
                      linewidth=0.3, alpha=0.7, label=f'{label} (r={r_val:.2f})', zorder=5)
    ax.plot([0.01, 100], [0.01, 100], '--', color='grey', linewidth=1, zorder=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Zeng Reference (%)', fontsize=12)
    ax.set_ylabel('Query platform (%)', fontsize=12)
    ax.set_title('All platforms vs Reference\n(class level)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')

    plt.tight_layout()
    plt.savefig(f'{OUT}/xenium_validation_proportions.png', dpi=250,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved {OUT}/xenium_validation_proportions.png")


# ════════════════════════════════════════════════════════════════════════
# FIGURE 3: DE concordant forest plot (meta FDR<0.2 + concordant ≥2)
# ════════════════════════════════════════════════════════════════════════
def plot_de_concordant():
    print("Figure 3: DE concordant forest + heatmap...")

    meta = pd.read_csv('output/de/de_meta_PREG_vs_CTRL.csv')

    # Load per-platform DE for SE estimation
    plat_data = {}
    for name, fname in [('merfish', 'de_merfish_PREG_vs_CTRL.csv'),
                         ('slidetags', 'de_slidetags_PREG_vs_CTRL.csv'),
                         ('xenium5k', 'de_xenium5k_PREG_vs_CTRL.csv')]:
        path = f'output/de/{fname}'
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['key'] = df['gene'] + '|' + df['celltype']
            plat_data[name] = df.set_index('key')

    # Filter: meta FDR<0.2 + concordant in ≥2 platforms
    def check(row):
        logfcs = {p: row[f'logFC_{p}'] for p in PLATFORMS if pd.notna(row.get(f'logFC_{p}'))}
        if len(logfcs) < 2:
            return False
        from collections import Counter
        signs = [np.sign(v) for v in logfcs.values()]
        return Counter(signs).most_common(1)[0][1] >= 2

    hits = meta[meta.fdr_combined < 0.20].copy()
    hits = hits[hits.apply(check, axis=1)].copy()
    hits = hits.sort_values('p_combined').reset_index(drop=True)

    # Biological grouping
    def bio_group(gene, ct):
        ct_l = ct.lower()
        if gene == 'Prl':
            return ('A', 'Prolactin signaling')
        if gene == 'Apoe':
            return ('B', 'Apoe / lipid transport')
        if gene in ('Cxcl13', 'Ccnd3'):
            return ('C', 'Immune / chemokine')
        if '060 ot d3 folh1' in ct_l:
            return ('D', 'OT D3 Folh1 Gaba')
        if '069 lsx nkx2-1' in ct_l:
            return ('E', 'LSX Nkx2-1 Gaba')
        if any(k in ct_l for k in ['glut']):
            return ('F', 'Other glutamatergic')
        if any(k in ct_l for k in ['gaba']):
            return ('G', 'Other GABAergic')
        return ('H', 'Non-neuronal')

    hits[['bio_key', 'bio_label']] = hits.apply(
        lambda r: pd.Series(bio_group(r['gene'], r['celltype'])), axis=1)
    hits = hits.sort_values(['bio_key', 'mean_logFC'], ascending=[True, False]).reset_index(drop=True)

    def short_ct(ct):
        parts = ct.split(' ', 1)
        return parts[1][:38] if len(parts) == 2 else ct[:38]

    hits['label'] = hits['gene'] + ' — ' + hits['celltype'].apply(short_ct)
    n = len(hits)

    # Save table
    out_cols = ['gene', 'celltype', 'bio_label', 'mean_logFC', 'fdr_combined',
                'n_platforms'] + [f'{m}_{p}' for p in PLATFORMS for m in ['logFC', 'pval', 'fdr']]
    hits[[c for c in out_cols if c in hits.columns]].to_csv(
        f'{OUT}/../de_concordant_hits_fdr02.csv', index=False)

    # ── Forest plot ────────────────────────────────────────────────────
    fig_h = max(10, n * 0.52 + 3)
    fig, ax = plt.subplots(figsize=(16, fig_h))

    cat_bg = {'A': '#FFF3E0', 'B': '#E8F5E9', 'C': '#FCE4EC',
              'D': '#EDE7F6', 'E': '#E3F2FD', 'F': '#FFF8E1',
              'G': '#E0F7FA', 'H': '#F3E5F5'}

    # Shading per category
    prev_key = None
    band_start = 0
    for i, (_, row) in enumerate(hits.iterrows()):
        if row['bio_key'] != prev_key:
            if prev_key is not None:
                rect = FancyBboxPatch((-9, band_start - 0.5), 18, i - band_start,
                    boxstyle="round,pad=0.05",
                    facecolor=cat_bg.get(prev_key, '#fafafa'), edgecolor='none', alpha=0.35, zorder=0)
                ax.add_patch(rect)
                mid = (band_start + i) / 2
                ax.text(-8.5, mid, hits.iloc[band_start]['bio_label'],
                        fontsize=12, fontstyle='italic', color='#555', va='center', ha='left', zorder=1)
            band_start = i
            prev_key = row['bio_key']
    # Last band
    rect = FancyBboxPatch((-9, band_start - 0.5), 18, n - band_start,
        boxstyle="round,pad=0.05",
        facecolor=cat_bg.get(prev_key, '#fafafa'), edgecolor='none', alpha=0.35, zorder=0)
    ax.add_patch(rect)
    ax.text(-8.5, (band_start + n) / 2, hits.iloc[band_start]['bio_label'],
            fontsize=12, fontstyle='italic', color='#555', va='center', ha='left', zorder=1)

    PLAT_OFFSETS = {'merfish': -0.22, 'slidetags': 0.0, 'xenium5k': 0.22}

    for i, (_, row) in enumerate(hits.iterrows()):
        for plat in PLATFORMS:
            lfc = row.get(f'logFC_{plat}')
            fdr = row.get(f'fdr_{plat}')
            pval = row.get(f'pval_{plat}')
            if pd.isna(lfc):
                continue

            y = i + PLAT_OFFSETS[plat]
            color = PLAT_COLORS[plat]

            if pd.notna(fdr) and fdr < 0.05:
                marker, ms = 's', 9
            elif pd.notna(fdr) and fdr < 0.1:
                marker, ms = 's', 7
            elif pd.notna(pval) and pval < 0.05:
                marker, ms = 'o', 6
            else:
                marker, ms = 'o', 4

            ax.plot(lfc, y, marker, color=color, markersize=ms, zorder=5,
                    markeredgecolor='black' if (pd.notna(fdr) and fdr < 0.1) else 'none',
                    markeredgewidth=0.8)

            # SE from F statistic
            key = f'{row["gene"]}|{row["celltype"]}'
            if plat in plat_data and key in plat_data[plat].index:
                pr = plat_data[plat].loc[key]
                if isinstance(pr, pd.DataFrame):
                    pr = pr.iloc[0]
                f_val = pr.get('F', np.nan)
                if pd.notna(f_val) and f_val > 0:
                    se = abs(lfc) / np.sqrt(f_val)
                    ci = 1.96 * se
                    ax.plot([lfc - ci, lfc + ci], [y, y], '-', color=color,
                            linewidth=1.2, alpha=0.4, zorder=4)

        # Meta diamond
        ax.plot(row['mean_logFC'], i, 'D', color='black', markersize=4.5,
                zorder=6, alpha=0.7)

    ax.axvline(0, color='grey', linewidth=0.8, linestyle='--', zorder=2)
    ax.set_yticks(range(n))
    ax.set_yticklabels(hits['label'], fontsize=12)
    ax.set_xlabel('log₂FC (PREG vs CTRL)', fontsize=16)
    ax.set_title('Cross-platform concordant DE (PREG vs CTRL)\n'
                 'Meta-analysis FDR < 0.2, concordant direction in ≥2 platforms',
                 fontsize=16, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlim(-9, 7)

    # Right margin: meta FDR
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(range(n))
    fdr_labels = []
    for _, row in hits.iterrows():
        fdr = row['fdr_combined']
        star = '***' if fdr < 0.01 else '**' if fdr < 0.05 else '*' if fdr < 0.1 else ''
        fdr_labels.append(f'FDR={fdr:.3f} {star}')
    ax2.set_yticklabels(fdr_labels, fontsize=10, color='#666', family='monospace')
    ax2.tick_params(axis='y', length=0)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PLAT_COLORS['merfish'],
               markersize=8, label='MERFISH'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PLAT_COLORS['slidetags'],
               markersize=8, label='Slide-tags'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PLAT_COLORS['xenium5k'],
               markersize=8, label='Xenium 5k'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='grey',
               markeredgecolor='black', markeredgewidth=0.8, markersize=8,
               label='FDR < 0.1'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='black',
               markersize=5, label='Meta mean'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11,
              framealpha=0.95, edgecolor='#ccc')

    plt.tight_layout()
    plt.savefig(f'{OUT}/de_concordant_forest_fdr02.png', dpi=250,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved {OUT}/de_concordant_forest_fdr02.png")

    # ── Heatmap ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, fig_h * 0.85),
                              gridspec_kw={'width_ratios': [3, 1.2], 'wspace': 0.05})
    ax_heat, ax_bar = axes

    lfc_mat = np.full((n, 3), np.nan)
    for i, (_, row) in enumerate(hits.iterrows()):
        for j, plat in enumerate(PLATFORMS):
            lfc_mat[i, j] = row.get(f'logFC_{plat}', np.nan)

    vmax = min(np.nanmax(np.abs(lfc_mat)), 5)
    im = ax_heat.imshow(lfc_mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')

    for i, (_, row) in enumerate(hits.iterrows()):
        for j, plat in enumerate(PLATFORMS):
            lfc = row.get(f'logFC_{plat}')
            fdr = row.get(f'fdr_{plat}')
            pval = row.get(f'pval_{plat}')
            if pd.isna(lfc):
                ax_heat.text(j, i, '—', ha='center', va='center', fontsize=10, color='#999')
                continue
            if pd.notna(fdr) and fdr < 0.05:
                txt, fw = f'{lfc:+.1f}\nFDR<.05', 'bold'
            elif pd.notna(fdr) and fdr < 0.1:
                txt, fw = f'{lfc:+.1f}\nFDR<.10', 'bold'
            elif pd.notna(pval) and pval < 0.05:
                txt, fw = f'{lfc:+.1f}\np<.05', 'normal'
            else:
                txt, fw = f'{lfc:+.1f}', 'normal'
            tc = 'white' if abs(lfc) > vmax * 0.55 else '#222'
            ax_heat.text(j, i, txt, ha='center', va='center', fontsize=9,
                         fontweight=fw, color=tc, linespacing=1.1)

    ax_heat.set_xticks(range(3))
    ax_heat.set_xticklabels(['MERFISH', 'Slide-tags', 'Xenium 5k'],
                             fontsize=14, fontweight='bold')
    ax_heat.set_yticks(range(n))
    ax_heat.set_yticklabels(hits['label'], fontsize=11)

    cbar = plt.colorbar(im, ax=ax_heat, shrink=0.3, pad=0.02, location='top')
    cbar.set_label('log₂FC', fontsize=12)

    ax_bar.barh(range(n), hits['mean_logFC'],
                color=['#E63946' if lfc > 0 else '#457B9D' for lfc in hits['mean_logFC']],
                height=0.7, alpha=0.8, edgecolor='white', linewidth=0.5)
    ax_bar.axvline(0, color='grey', linewidth=0.8)
    ax_bar.set_yticks([])
    ax_bar.set_xlabel('Meta logFC', fontsize=12)
    ax_bar.invert_yaxis()

    for i, (_, row) in enumerate(hits.iterrows()):
        fdr = row['fdr_combined']
        star = '***' if fdr < 0.01 else '**' if fdr < 0.05 else '*' if fdr < 0.1 else ''
        lfc = row['mean_logFC']
        offset = 0.15 if lfc > 0 else -0.15
        ha = 'left' if lfc > 0 else 'right'
        ax_bar.text(lfc + offset, i, star, fontsize=10, va='center', ha=ha,
                    fontweight='bold', color='#333')

    ax_bar.set_title('Meta\n*<.10 **<.05 ***<.01', fontsize=10, color='#555')

    fig.suptitle('Per-platform logFC for concordant DE hits (meta FDR < 0.2)',
                 fontsize=16, fontweight='bold', y=1.01)

    plt.tight_layout()
    plt.savefig(f'{OUT}/de_concordant_heatmap_fdr02.png', dpi=250,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved {OUT}/de_concordant_heatmap_fdr02.png")

    return hits


# ════════════════════════════════════════════════════════════════════════
# FIGURE 4: DE overview stats
# ════════════════════════════════════════════════════════════════════════
def plot_de_overview():
    print("Figure 4: DE overview stats...")

    meta = pd.read_csv('output/de/de_meta_PREG_vs_CTRL.csv')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Tests per platform
    ax = axes[0]
    for j, plat in enumerate(PLATFORMS):
        col = f'logFC_{plat}'
        n_tested = meta[col].notna().sum()
        n_fdr05 = (meta[f'fdr_{plat}'] < 0.05).sum() if f'fdr_{plat}' in meta.columns else 0
        n_fdr10 = (meta[f'fdr_{plat}'] < 0.10).sum() if f'fdr_{plat}' in meta.columns else 0

        ax.bar(j, n_tested, color=PLAT_COLORS[plat], alpha=0.3, width=0.6,
              label=f'{PLAT_LABELS[plat]} tested' if j == 0 else None)
        ax.text(j, n_tested + 500, f'{n_tested:,}', ha='center', fontsize=10,
                fontweight='bold')
        ax.text(j, -2500, f'FDR<.05: {n_fdr05}\nFDR<.10: {n_fdr10}',
                ha='center', fontsize=9, color='#555')

    ax.set_xticks(range(3))
    ax.set_xticklabels([PLAT_LABELS[p] for p in PLATFORMS], fontsize=12)
    ax.set_ylabel('Gene × celltype tests', fontsize=12)
    ax.set_title('Per-platform DE testing volume', fontsize=13, fontweight='bold')

    # Panel B: Meta-analysis FDR distribution
    ax = axes[1]
    fdr = meta['fdr_combined'].dropna()
    thresholds = [0.01, 0.05, 0.10, 0.20, 0.50]
    counts = [int((fdr < t).sum()) for t in thresholds]
    bars = ax.bar(range(len(thresholds)), counts, color='#2A9D8F', alpha=0.7,
                  width=0.6, edgecolor='white')
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f'<{t}' for t in thresholds], fontsize=11)
    ax.set_xlabel('Meta-analysis FDR threshold', fontsize=12)
    ax.set_ylabel('Number of hits', fontsize=12)
    ax.set_title(f'Meta-analysis FDR distribution\n({len(fdr):,} total gene×celltype tests)',
                 fontsize=13, fontweight='bold')
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, cnt + 20, str(cnt),
                ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUT}/de_overview_stats.png', dpi=250,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved {OUT}/de_overview_stats.png")


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    plot_crumblr_boxplots()
    plot_xenium_validation()
    hits = plot_de_concordant()
    plot_de_overview()
    print(f"\nAll figures saved to {OUT}/")
