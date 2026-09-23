"""Show camera improvement and rod outcomes side by side, including refusals."""
from __future__ import annotations

import numpy as np
from PIL import Image
from run_rod_reference import ROOT, digest, read_json, write_json


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path = ROOT / 'docs/experiments/results/2026-09-23-rod-reference.json'
    report = read_json(path)
    names = ['r01', 'r02', 'r03', 'r04']
    labels = ['Complete / fixed', 'Gap / fixed', 'Thin / fixed', 'Gap / zoom']
    variants = ['initial', 'common_K_focal_linear', 'common_K_focal_robust', 'common_K_focal_trimmed']
    colors = ['#888888', '#bd7828', '#5c91b6', '#17699b']
    short = ['Initial', 'Linear', 'Robust', 'Two-stage']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    x = np.arange(4)
    for i, variant in enumerate(variants):
        camera = [next(r for r in report['cameras'] if r['case_id'] == cid and r['variant'] == variant) for cid in names]
        offset = (i - 1.5) * .19
        for ax, values in ((axes[0, 0], [r['camera']['camera_rmse_m'] * 1000 for r in camera]),
                           (axes[0, 1], [r['points']['summary']['median'] * 1000 for r in camera])):
            ax.bar(x + offset, values, width=.18, color=colors[i], label=short[i])
        for ax, method in ((axes[1, 0], 'baseline'), (axes[1, 1], 'cylinder_support')):
            rows = [next(r for r in report['rods'] if r['case_id'] == cid and r['camera_variant'] == variant and r['method'] == method) for cid in names]
            values = [100 * (r['metrics']['recovery_fraction'] or 0) for r in rows]
            ax.bar(x + offset, values, width=.18, color=colors[i])
            for j, row in enumerate(rows):
                if row['state'] != 'accepted':
                    ax.plot(j + offset, 1, 'x', color=colors[i], markersize=6)
    axes[0, 0].set(ylabel='Camera-center RMSE (mm)', yscale='log')
    axes[0, 1].set(ylabel='Held-out point median error (mm)', yscale='log')
    axes[1, 0].set(ylabel='Ordinary rod: recovered length (%)', ylim=(-4, 108))
    axes[1, 1].set(ylabel='Cylinder-screened rod: recovered length (%)', ylim=(-4, 108))
    for ax in axes.ravel():
        ax.set(xticks=x, xticklabels=labels)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle('Reference-assisted rod transfer | 4 synthetic conditions, 3 distinct rod layouts', fontsize=14)
    fig.supxlabel('Camera-only Sim3 for GT scoring. Rod tolerance: 25 mm. x = withheld/no output. No dense patch applied.', fontsize=9)
    figure = ROOT / 'docs/experiments/figures/2026-09-23-rod-reference.png'
    fig.savefig(figure, dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), layout='constrained')
    for ax, cid, label in zip(axes, names, labels):
        ax.imshow(Image.open(ROOT / 'data/inputs/rod-reference-scenes-v1-20260923' / cid / 'view_+03.png'))
        ax.set_title(cid + ' ' + label)
        ax.axis('off')
    fig.suptitle('New Blender RGB | textured reference panels are an explicit acquisition aid')
    overview = ROOT / 'docs/experiments/figures/2026-09-23-rod-reference-rgb.png'
    fig.savefig(overview, dpi=150)
    plt.close(fig)
    write_json(ROOT / 'docs/experiments/results/2026-09-23-rod-reference-figures.json', dict(summary_sha256=digest(path),
        plotting_source_sha256=digest(__file__), figures={p.name: digest(p) for p in (figure, overview)}))


if __name__ == '__main__':
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    plot()
