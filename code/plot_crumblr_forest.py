#!/usr/bin/env python3
"""
Crumblr compositional analysis forest plot (PREG vs CTRL).

Shows per-platform logFC ± 95% CI for cell type proportion changes,
organized into non-neuronal and neuronal panels, with endothelial cells highlighted.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = 'output/analysis_summary/figures'
os.makedirs(OUT, exist_ok=True)

PLATFORMS = ['merfish', 'slidetags', 'xenium5k']
PLAT_LABELS = {'merfish': 'MERFISH', 'slidetags': 'Slide-tags', 'xenium5k': 'Xenium 5k'}
PLAT_COLORS = {'merfish': '#E63946', 'slidetags': '#457B9D', 'xenium5k': '#2A9D8F'}
PLAT_OFFSETS = {'merfish': -0.24, 'slidetags': 0.0, 'xenium5k': 0.24}

NEURONAL_KW = ('Glut', 'GABA', 'Gaba', 'Dopa', 'Sero', 'Gnrh1')

# Platform sample sizes (for Stouffer weights)
PLAT_N_ANIMALS = {'merfish': 9, 'slidetags': 8, 'xenium5k': 6}

# ── Load data ──────────────────────────────────────────────────────────
cr = pd.read_csv('output/crumblr/crumblr_results_all.csv')
preg = cr[cr.contrast == 'PREG_vs_CTRL'].copy()

# Use hier_subclass level for cross-platform comparison
levels = {p: f'{p}_hier_subclass' for p in PLATFORMS}
sub = preg[preg.level.isin(levels.values())].copy()
sub['platform'] = sub['level'].str.split('_').str[0]

# Build per-celltype, per-platform data
ct_data = {}
for ct in sub.celltype.unique():
    rows = sub[sub.celltype == ct]
    entry = {}
    for _, r in rows.iterrows():
        entry[r['platform']] = {
            'logFC': r['logFC'], 'SE': r['SE'],
            'P.Value': r['P.Value'], 'FDR': r['FDR'], 't': r['t']
        }
    if len(entry) >= 2:
        ct_data[ct] = entry

# ── Meta-analysis (Stouffer's weighted Z) ──────────────────────────────
from scipy.stats import norm

def meta_analyze_crumblr(entry):
    """Stouffer's weighted Z-score meta-analysis across platforms."""
    z_scores = []
    weights = []
    logfcs = []
    ses = []
    for p in PLATFORMS:
        if p not in entry:
            continue
        d = entry[p]
        pval = d['P.Value']
        lfc = d['logFC']
        # Directional Z-score
        z = norm.ppf(1 - pval / 2)  # two-sided p to |Z|
        if lfc < 0:
            z = -z
        w = np.sqrt(PLAT_N_ANIMALS[p])
        z_scores.append(z)
        weights.append(w)
        logfcs.append(lfc)
        ses.append(d['SE'])

    if len(z_scores) < 2:
        return None

    z_arr = np.array(z_scores)
    w_arr = np.array(weights)
    z_combined = np.sum(w_arr * z_arr) / np.sqrt(np.sum(w_arr ** 2))
    p_combined = 2 * (1 - norm.cdf(abs(z_combined)))  # two-sided

    # Inverse-variance weighted mean logFC
    inv_var = np.array([1 / (se ** 2) for se in ses])
    mean_logFC = np.sum(np.array(logfcs) * inv_var) / np.sum(inv_var)
    se_meta = 1 / np.sqrt(np.sum(inv_var))

    return {
        'z_combined': z_combined,
        'p_combined': p_combined,
        'mean_logFC': mean_logFC,
        'se_meta': se_meta,
        'n_platforms': len(z_scores),
    }

# Add meta-analysis to each cell type
for ct in ct_data:
    ct_data[ct]['_meta'] = meta_analyze_crumblr(ct_data[ct])

# FDR correction on meta p-values
from statsmodels.stats.multitest import multipletests
cts_with_meta = [ct for ct in ct_data if ct_data[ct]['_meta'] is not None]
meta_pvals = np.array([ct_data[ct]['_meta']['p_combined'] for ct in cts_with_meta])
_, fdr_vals, _, _ = multipletests(meta_pvals, method='fdr_bh')
for ct, fdr in zip(cts_with_meta, fdr_vals):
    ct_data[ct]['_meta']['fdr_combined'] = fdr

print(f"\nMeta-analysis FDR<0.05: {(fdr_vals < 0.05).sum()}")
print(f"Meta-analysis FDR<0.10: {(fdr_vals < 0.10).sum()}")
print(f"Meta-analysis FDR<0.20: {(fdr_vals < 0.20).sum()}")

def is_neuronal(ct):
    return any(kw in ct for kw in NEURONAL_KW)

def short_ct(ct):
    parts = ct.split(' ', 1)
    return parts[1][:42] if len(parts) == 2 else ct[:42]

# ── Select cell types ──────────────────────────────────────────────────
def meta_p(ct):
    m = ct_data[ct].get('_meta')
    return m['p_combined'] if m else 1.0

# Non-neuronal: show ALL (there aren't that many, and Endo is here)
nn_cts = sorted([ct for ct in ct_data if not is_neuronal(ct)], key=meta_p)

# Neuronal: top 25 by meta p-value
neur_cts = sorted([ct for ct in ct_data if is_neuronal(ct)], key=meta_p)
neur_cts = neur_cts[:25]

print(f"Non-neuronal cell types: {len(nn_cts)}")
print(f"Neuronal cell types (top 25): {len(neur_cts)}")
endo_in = any('Endo' in ct for ct in nn_cts)
print(f"Endo in non-neuronal panel: {endo_in}")


def draw_forest(ax, ct_list, title, highlight_cts=None):
    """Draw a forest plot panel."""
    if highlight_cts is None:
        highlight_cts = []

    n = len(ct_list)

    for i, ct in enumerate(ct_list):
        is_highlight = any(h in ct for h in highlight_cts)

        # Highlight background
        if is_highlight:
            rect = Rectangle((-4, i - 0.45), 8, 0.9,
                            facecolor='#FFF3E0', edgecolor='#FF9800',
                            linewidth=1.5, alpha=0.5, zorder=0)
            ax.add_patch(rect)

        entry = ct_data[ct]

        for plat in PLATFORMS:
            if plat not in entry:
                continue

            d = entry[plat]
            y = i + PLAT_OFFSETS[plat]
            color = PLAT_COLORS[plat]
            lfc = d['logFC']
            se = d['SE']
            pval = d['P.Value']
            fdr = d['FDR']

            # CI
            ci_lo = lfc - 1.96 * se
            ci_hi = lfc + 1.96 * se
            ax.plot([ci_lo, ci_hi], [y, y], '-', color=color,
                    linewidth=1.8, alpha=0.45, zorder=3)

            # Marker size/shape by significance
            if fdr < 0.05:
                marker, ms, edge = 'D', 8, 'black'
            elif fdr < 0.10:
                marker, ms, edge = 's', 7, 'black'
            elif pval < 0.05:
                marker, ms, edge = 'o', 6, '#555'
            else:
                marker, ms, edge = 'o', 4.5, 'none'

            ax.plot(lfc, y, marker, color=color, markersize=ms, zorder=5,
                    markeredgecolor=edge, markeredgewidth=0.8 if edge != 'none' else 0)

        # Right-side concordance annotation
        signs = [np.sign(entry[p]['logFC']) for p in PLATFORMS if p in entry]
        n_plat = len(signs)
        n_pos = sum(1 for s in signs if s > 0)
        n_neg = sum(1 for s in signs if s < 0)
        if n_pos == n_plat:
            conc_txt = f'↑ {n_plat}/{n_plat}'
            conc_color = '#C62828'
        elif n_neg == n_plat:
            conc_txt = f'↓ {n_plat}/{n_plat}'
            conc_color = '#1565C0'
        else:
            conc_txt = f'mixed'
            conc_color = '#888'

        # Meta-analysis diamond
        meta_info = entry.get('_meta')
        if meta_info:
            m_lfc = meta_info['mean_logFC']
            m_se = meta_info['se_meta']
            m_fdr = meta_info.get('fdr_combined', 1.0)
            # Diamond
            ax.plot(m_lfc, i, 'D', color='black', markersize=6, zorder=6, alpha=0.8)
            # Meta CI
            ax.plot([m_lfc - 1.96 * m_se, m_lfc + 1.96 * m_se], [i, i],
                    '-', color='black', linewidth=2.5, alpha=0.3, zorder=4)

        # Meta FDR annotation
        meta_fdr_txt = ''
        if meta_info:
            mf = meta_info.get('fdr_combined', 1.0)
            star = '***' if mf < 0.01 else '**' if mf < 0.05 else '*' if mf < 0.10 else ''
            meta_fdr_txt = f' {star}' if star else ''

        ax.text(4.1, i, f'{conc_txt}{meta_fdr_txt}', fontsize=7.5, va='center', ha='left',
                color=conc_color, fontweight='bold' if n_pos == n_plat or n_neg == n_plat else 'normal',
                family='monospace')

    ax.axvline(0, color='grey', linewidth=0.8, linestyle='--', zorder=2)
    ax.set_yticks(range(n))
    labels = []
    for ct in ct_list:
        s = short_ct(ct)
        if any(h in ct for h in (highlight_cts or [])):
            s = '★ ' + s
        labels.append(s)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('logFC (PREG vs CTRL)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlim(-4, 4)

    # Bold endothelial y-tick labels
    for i, ct in enumerate(ct_list):
        if any(h in ct for h in (highlight_cts or [])):
            ax.get_yticklabels()[i].set_fontweight('bold')
            ax.get_yticklabels()[i].set_color('#E65100')


# ════════════════════════════════════════════════════════════════════════
# Figure: Two-panel forest plot
# ════════════════════════════════════════════════════════════════════════
fig_h_nn = max(5, len(nn_cts) * 0.42 + 2)
fig_h_ne = max(5, len(neur_cts) * 0.42 + 2)
fig_h = max(fig_h_nn, fig_h_ne)

fig, axes = plt.subplots(1, 2, figsize=(20, fig_h),
                          gridspec_kw={'width_ratios': [1, 1.2], 'wspace': 0.45})

draw_forest(axes[0], nn_cts,
            f'Non-neuronal cell types (n={len(nn_cts)})',
            highlight_cts=['333 Endo'])

draw_forest(axes[1], neur_cts,
            f'Top neuronal cell types (n={len(neur_cts)})',
            highlight_cts=[])

# Shared legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=PLAT_COLORS['merfish'],
           markersize=8, label='MERFISH (n=9 samples)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=PLAT_COLORS['slidetags'],
           markersize=8, label='Slide-tags (n=8 samples)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=PLAT_COLORS['xenium5k'],
           markersize=8, label='Xenium 5k (n=12 samples)'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='grey',
           markeredgecolor='black', markeredgewidth=0.8, markersize=7,
           label='FDR < 0.05'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='grey',
           markeredgecolor='black', markeredgewidth=0.8, markersize=7,
           label='FDR < 0.10'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='grey',
           markeredgecolor='#555', markeredgewidth=0.8, markersize=6,
           label='p < 0.05 (nom.)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='grey',
           markeredgecolor='none', markersize=5, label='p ≥ 0.05'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='black',
           markersize=6, label='Meta-analysis mean'),
    Line2D([0], [0], color='grey', linewidth=1.5, alpha=0.5,
           label='95% CI'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=5,
           fontsize=10, framealpha=0.95, edgecolor='#ccc',
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle('crumblr compositional analysis: cell type proportion changes in pregnancy (PREG vs CTRL)\n'
             'Per-platform logFC ± 95% CI from ALR-transformed dream models\n'
             'Right margin: direction concordance across platforms (↑ up / ↓ down / mixed)',
             fontsize=14, fontweight='bold', y=1.03)

plt.savefig(f'{OUT}/crumblr_forest_all.png', dpi=250,
            bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved {OUT}/crumblr_forest_all.png")


# ════════════════════════════════════════════════════════════════════════
# Focused non-neuronal panel (larger, clearer)
# ════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, max(6, len(nn_cts) * 0.55 + 2)))

n = len(nn_cts)
for i, ct in enumerate(nn_cts):
    is_endo = '333 Endo' in ct

    if is_endo:
        rect = Rectangle((-4.5, i - 0.45), 9.5, 0.9,
                         facecolor='#FFF3E0', edgecolor='#FF9800',
                         linewidth=2, alpha=0.6, zorder=0)
        ax.add_patch(rect)

    entry = ct_data[ct]

    for plat in PLATFORMS:
        if plat not in entry:
            continue

        d = entry[plat]
        y = i + PLAT_OFFSETS[plat]
        color = PLAT_COLORS[plat]
        lfc = d['logFC']
        se = d['SE']
        pval = d['P.Value']
        fdr = d['FDR']

        ci_lo = lfc - 1.96 * se
        ci_hi = lfc + 1.96 * se
        ax.plot([ci_lo, ci_hi], [y, y], '-', color=color,
                linewidth=2, alpha=0.5, zorder=3)

        if fdr < 0.05:
            marker, ms, edge = 'D', 9, 'black'
        elif fdr < 0.10:
            marker, ms, edge = 's', 8, 'black'
        elif pval < 0.05:
            marker, ms, edge = 'o', 7, '#555'
        else:
            marker, ms, edge = 'o', 5, 'none'

        ax.plot(lfc, y, marker, color=color, markersize=ms, zorder=5,
                markeredgecolor=edge, markeredgewidth=1 if edge != 'none' else 0)

    # Meta diamond
    meta_info = entry.get('_meta')
    if meta_info:
        m_lfc = meta_info['mean_logFC']
        m_se = meta_info['se_meta']
        ax.plot(m_lfc, i, 'D', color='black', markersize=7, zorder=6, alpha=0.8)
        ax.plot([m_lfc - 1.96 * m_se, m_lfc + 1.96 * m_se], [i, i],
                '-', color='black', linewidth=2.5, alpha=0.3, zorder=4)

ax.axvline(0, color='grey', linewidth=0.8, linestyle='--', zorder=2)
ax.set_yticks(range(n))

labels = []
for ct in nn_cts:
    s = short_ct(ct)
    labels.append(s)
ax.set_yticklabels(labels, fontsize=11)
ax.invert_yaxis()
ax.set_xlabel('logFC (PREG vs CTRL)', fontsize=13)
ax.set_xlim(-4.5, 5)

# Highlight Endo label
for i, ct in enumerate(nn_cts):
    if '333 Endo' in ct:
        ax.get_yticklabels()[i].set_fontweight('bold')
        ax.get_yticklabels()[i].set_color('#E65100')
        ax.get_yticklabels()[i].set_fontsize(12)

# Right annotation: per-platform p-value
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(range(n))
right_labels = []
for ct in nn_cts:
    entry = ct_data[ct]
    parts = []
    for p in PLATFORMS:
        if p in entry:
            pv = entry[p]['P.Value']
            fdr = entry[p]['FDR']
            lfc = entry[p]['logFC']
            star = '**' if fdr < 0.05 else '*' if fdr < 0.10 else '·' if pv < 0.05 else ' '
            parts.append(f'{p[:2]}:{lfc:+.2f}{star}')
        else:
            parts.append(f'{p[:2]}:  —  ')
    meta_info = entry.get('_meta')
    if meta_info:
        mf = meta_info.get('fdr_combined', 1.0)
        meta_star = '***' if mf < 0.01 else '**' if mf < 0.05 else '*' if mf < 0.10 else ''
        parts.append(f'meta:{meta_info["mean_logFC"]:+.2f} FDR={mf:.3f}{meta_star}')
    right_labels.append('  '.join(parts))

ax2.set_yticklabels(right_labels, fontsize=8, family='monospace', color='#555')
ax2.tick_params(axis='y', length=0)

ax.set_title('Non-neuronal cell type proportion changes in pregnancy (crumblr)\n'
             'Per-platform logFC ± 95% CI | ★ = Endothelial cells highlighted\n'
             '** FDR<0.05  * FDR<0.10  · nom. p<0.05',
             fontsize=13, fontweight='bold')

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=PLAT_COLORS['merfish'],
           markersize=9, label='MERFISH'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=PLAT_COLORS['slidetags'],
           markersize=9, label='Slide-tags'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=PLAT_COLORS['xenium5k'],
           markersize=9, label='Xenium 5k'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='grey',
           markeredgecolor='black', markeredgewidth=0.8, markersize=8,
           label='FDR < 0.05'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='grey',
           markeredgecolor='black', markeredgewidth=0.8, markersize=8,
           label='FDR < 0.10'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='black',
           markersize=7, label='Meta-analysis mean'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10,
          framealpha=0.95, edgecolor='#ccc')

plt.tight_layout()
plt.savefig(f'{OUT}/crumblr_forest_nonneuronal.png', dpi=250,
            bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved {OUT}/crumblr_forest_nonneuronal.png")


# ── Print Endo summary ────────────────────────────────────────────────
print("\n=== Endothelial (333 Endo NN) crumblr summary ===")
endo = ct_data.get('333 Endo NN', {})
for p in PLATFORMS:
    if p in endo:
        d = endo[p]
        print(f"  {PLAT_LABELS[p]:12s}: logFC={d['logFC']:+.3f} ± {d['SE']:.3f}, "
              f"p={d['P.Value']:.3e}, FDR={d['FDR']:.3f}")
    else:
        print(f"  {PLAT_LABELS[p]:12s}: not tested")

# All 3 platforms show positive logFC — concordant direction but none significant
signs = [np.sign(endo[p]['logFC']) for p in PLATFORMS if p in endo]
print(f"  Direction concordance: {'all positive' if all(s > 0 for s in signs) else 'mixed'}")
print(f"  Best platform p: {min(endo[p]['P.Value'] for p in PLATFORMS if p in endo):.4f}")
m = endo.get('_meta')
if m:
    print(f"  Meta p={m['p_combined']:.4f}, FDR={m.get('fdr_combined', np.nan):.4f}, logFC={m['mean_logFC']:+.3f}")
print(f"  → Concordant positive direction but not significant in any platform or meta-analysis")

# ── Save meta-analysis table ──────────────────────────────────────────
rows = []
for ct in sorted(ct_data.keys()):
    m = ct_data[ct].get('_meta')
    if m is None:
        continue
    row = {
        'celltype': ct,
        'neuronal': is_neuronal(ct),
        'meta_logFC': m['mean_logFC'],
        'meta_SE': m['se_meta'],
        'meta_z': m['z_combined'],
        'meta_p': m['p_combined'],
        'meta_fdr': m.get('fdr_combined', np.nan),
        'n_platforms': m['n_platforms'],
    }
    for p in PLATFORMS:
        if p in ct_data[ct] and p != '_meta':
            row[f'logFC_{p}'] = ct_data[ct][p]['logFC']
            row[f'SE_{p}'] = ct_data[ct][p]['SE']
            row[f'pval_{p}'] = ct_data[ct][p]['P.Value']
            row[f'fdr_{p}'] = ct_data[ct][p]['FDR']
    rows.append(row)

meta_df = pd.DataFrame(rows).sort_values('meta_p')
meta_df.to_csv('output/analysis_summary/crumblr_meta_analysis.csv', index=False)
print(f"\nSaved crumblr meta-analysis: output/analysis_summary/crumblr_meta_analysis.csv ({len(meta_df)} rows)")

print("\nDone!")
