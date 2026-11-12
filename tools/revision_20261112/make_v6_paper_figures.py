from __future__ import annotations

from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / 'revision_data'
DATA = REV / 'final_v6_ransac/paired_rerun_20261111'
OUT = REV / 'final_v6_ransac/figures'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REV / 'tools'))

MODELS = ['RandLA-Net', 'PointNet2-RGB', 'PointNeXt-L', 'PointGroup']
COLORS = {'RandLA-Net': '#1f77b4', 'PointNet2-RGB': '#d95f02', 'PointNeXt-L': '#2ca02c', 'PointGroup': '#9467bd'}
TRAITS = ['height_cm', 'crown_diameter_mean_cm', 'volume_alpha_cm3', 'compactness']
TRAIT_TITLES = {
    'height_cm': 'Plant height (cm)',
    'crown_diameter_mean_cm': 'Crown diameter (cm)',
    'volume_alpha_cm3': 'Alpha-shape volume (cm³)',
    'compactness': 'Compactness (-)',
}
TRAIT_LIMITS = {
    'height_cm': (-6.0, 9.0),
    'crown_diameter_mean_cm': (-40.0, 15.0),
    'volume_alpha_cm3': (-2800.0, 1000.0),
    'compactness': (-0.025, 0.012),
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.titlesize': 9.5,
    'axes.labelsize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': 0.7, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
})


def save(fig, stem: str) -> None:
    fig.savefig(OUT / f'{stem}.png', dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT / f'{stem}.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def stars(p: float) -> str:
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return ''


def load_pairs() -> pd.DataFrame:
    frame = pd.read_csv(DATA / 'v6_paired_trait_deltas.csv')
    frame['delta_abs_error_adpv_minus_base'] = frame['adpv_abs_error'] - frame['base_abs_error']
    return frame


def paired_trait_figure() -> None:
    data = load_pairs()
    stats = pd.read_csv(DATA / 'v6_paired_trait_statistics.csv')
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), sharey=False)
    for ax, trait in zip(axes.flat, TRAITS):
        g = data[data['trait'] == trait]
        s = stats[stats['trait'] == trait].set_index('model')
        positions = {m: i for i, m in enumerate(reversed(MODELS))}
        for model in MODELS:
            d = g[g['model'] == model]['delta_abs_error_adpv_minus_base'].to_numpy(float)
            y0 = positions[model]
            jitter = np.linspace(-0.10, 0.10, len(d)) if len(d) else np.array([])
            ax.scatter(d, y0 + jitter, s=10, color=COLORS[model], alpha=0.42, linewidths=0)
            row = s.loc[model]
            mean = float(row['mean_delta_abs_error_adpv_minus_base'])
            lo = float(row['ci95_low']); hi = float(row['ci95_high'])
            ax.errorbar(mean, y0, xerr=[[mean - lo], [hi - mean]], fmt='o', ms=4.5,
                        color='black', ecolor='black', elinewidth=1.0, capsize=2.5, zorder=5)
            label = f'n={int(row["n"])} {stars(float(row["holm_adjusted_p"]))}'.strip()
            ax.text(0.985, y0 + 0.23, label, transform=ax.get_yaxis_transform(),
                    ha='right', va='bottom', fontsize=7, color='#333333')
        ax.axvline(0, color='#555555', lw=0.8, ls='--')
        ax.set_xlim(*TRAIT_LIMITS[trait])
        ax.set_yticks([positions[m] for m in reversed(MODELS)], list(reversed(MODELS)))
        ax.set_ylim(-0.65, 3.65)
        ax.set_xlabel('Δ absolute error (ADPV − Base; <0 better)')
        ax.set_title(TRAIT_TITLES[trait], loc='left', fontweight='bold')
        ax.grid(axis='x', color='#dddddd', lw=0.5); ax.set_axisbelow(True)
        for side in ['top', 'right']: ax.spines[side].set_visible(False)
    fig.text(0.5, 0.016, 'Same plants before/after refinement; horizontal bars show mean and bootstrap 95% CI. Holm-adjusted Wilcoxon: *p<0.05, **p<0.01, ***p<0.001.',
             ha='center', va='bottom', fontsize=7.0, color='#444444')
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    save(fig, 'fig7_paired_trait_delta')


def paired_identity_figure() -> None:
    data = load_pairs()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
    for ax, trait in zip(axes.flat, TRAITS):
        g = data[data['trait'] == trait]
        for model in MODELS:
            q = g[g['model'] == model]
            ax.scatter(q['base_abs_error'], q['adpv_abs_error'], s=11, color=COLORS[model],
                       alpha=0.45, linewidths=0, label=f'{model} (n={len(q)})')
        vals = pd.concat([g['base_abs_error'], g['adpv_abs_error']]).to_numpy(float)
        lo, hi = float(vals.min()), float(vals.max())
        ax.plot([lo, hi], [lo, hi], color='#444444', lw=0.9, ls='--')
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Base absolute error'); ax.set_ylabel('ADPV absolute error')
        ax.set_title(TRAIT_TITLES[trait], loc='left', fontweight='bold')
        ax.grid(color='#e5e5e5', lw=0.5); ax.set_axisbelow(True)
        for side in ['top', 'right']: ax.spines[side].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.995), ncol=4, fontsize=7.0, frameon=False)
    fig.text(0.5, 0.005, 'Points below the identity line improved after refinement. Only plants matched by both Base and ADPV are shown.',
             ha='center', va='bottom', fontsize=7.0, color='#444444')
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    save(fig, 'figS_paired_trait_identity')


def sensitivity_figure() -> None:
    src = REV / 'final_v6_ransac/figures'
    rerun_stem = 'fig8_sensitivity_trainval_rerun_20261111'
    if all((src / f'{rerun_stem}.{ext}').exists() for ext in ('png', 'pdf')):
        for ext in ('png', 'pdf'):
            shutil.copy2(src / f'{rerun_stem}.{ext}', OUT / f'fig8_sensitivity_trainval.{ext}')
    else:
        raise FileNotFoundError('Rerun sensitivity figure is missing')


def main() -> None:
    paired_trait_figure()
    paired_identity_figure()
    sensitivity_figure()
    import subprocess
    subprocess.run([sys.executable, str(REV / 'tools/plot_efficiency_20261111.py')], check=True)
    index = '''# Final paper figure paths (v6)\n\n- Figure 6 main: `fig6_main_v6.pdf/png`\n- Figure 6 full-scene annotation: `fig6_full_scene_annotation_v6.pdf/png`\n- Figure 7 paired trait delta: `fig7_paired_trait_delta.pdf/png` (trait pairing rerun 2026-11-11)\n- Figure 8 train+val sensitivity: `fig8_sensitivity_trainval.pdf/png` (rerun 2026-11-11)\n- Supplementary paired identity: `figS_paired_trait_identity.pdf/png` (trait pairing rerun 2026-11-11)\n- Figure 10 efficiency: `fig10_efficiency_fixed_test.pdf/png` (unified KNN-R rerun 2026-11-11).\n'''
    (OUT / 'PAPER_FIGURE_INDEX.md').write_text(index, encoding='utf-8')
    print('WROTE', OUT)
    print(index)


if __name__ == '__main__':
    main()

