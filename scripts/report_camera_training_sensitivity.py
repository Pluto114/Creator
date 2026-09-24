"""Readable paired camera sensitivity tables and a scatter plot, no new scoring."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_camera_bundle_pilot import ROOT, digest, read_json, write_json

SOURCE = ROOT / 'docs/experiments/results/2026-09-24-camera-training-sensitivity-r2.json'
OUTPUT = ROOT / 'docs/experiments/results/2026-09-24-camera-training-readable.json'
FIGURE = ROOT / 'docs/experiments/figures/2026-09-24-camera-training-sensitivity.png'


def ranges(values):
    values = [v for v in values if v is not None]
    return None if not values else dict(min=min(values), max=max(values), count=len(values))


def report(source, output, figure):
    data = read_json(source)
    assert data['state'] == 'complete'
    keys = sorted({(r['parent'], r['case_id']) for r in data['cameras']})
    cases = []
    for parent, cid in keys:
        cameras = [r for r in data['cameras'] if (r['parent'], r['case_id']) == (parent, cid)]
        camera_folds = [r for r in cameras if r['condition'] != 'control']
        comparisons = [r for r in data['comparisons'] if (r['parent'], r['case_id']) == (parent, cid)]
        to_control = [r for r in comparisons if r['left'] == 'control']
        methods = {}
        for method in ('baseline', 'cylinder_support'):
            rows = [r for r in data['rows'] if (r['parent'], r['case_id'], r['method']) == (parent, cid, method)]
            full = next(r for r in rows if r['condition'] == 'control')
            folds = [r for r in rows if r['condition'] != 'control']
            accepted = [r for r in folds if r['state'] == 'accepted']
            curve = [next(v for v in c['rods'] if v['method'] == method) for c in to_control]
            methods[method] = dict(control=full, fold_states=dict(Counter(r['state'] for r in folds)),
                fold_recovery=ranges([r['metrics']['recovery_fraction'] for r in accepted]),
                fold_precision=ranges([r['metrics']['precision_fraction'] for r in accepted]),
                fold_p95_m=ranges([r['metrics']['truth_to_prediction']['distance_p95'] for r in accepted]),
                control_to_fold_curve_p95_over_baseline=ranges([r.get('symmetric_p95_over_baseline') for r in curve]),
                comparison_reasons=[r.get('reason') for r in curve])
        cases.append(dict(parent=parent, case_id=cid, methods=methods,
            fold_camera_decisions=dict(Counter(r['decision']['state'] for r in camera_folds)),
            fold_all_view_median_px=ranges([r['all_view_residual']['median_px'] for r in camera_folds if r['all_view_residual']]),
            fold_all_view_p95_px=ranges([r['all_view_residual']['p95_px'] for r in camera_folds if r['all_view_residual']]),
            control_to_fold_camera_center_over_baseline=ranges([c['camera'].get('center_delta_over_baseline_max') for c in to_control]),
            control_to_fold_rotation_degrees=ranges([c['camera'].get('rotation_degrees_max') for c in to_control]),
            pairwise_center_over_baseline=ranges([c['camera'].get('center_delta_over_baseline_max') for c in comparisons])))
    write_json(output, dict(source=source.relative_to(ROOT).as_posix(), source_sha256=digest(source), cases=cases,
                            scope='Ranges of four correlated development folds, not confidence intervals. Accepted-only physical ranges explicitly count available rows.'))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    palette = {'r01': '#16836b', 'r02': '#db7c23', 'r03': '#7d58b5', 'r04': '#657583'}
    names = {'r01': 'complete', 'r02': 'gap', 'r03': 'thin', 'r04': 'zoom'}
    for ax, parent in zip(axes, ['rod-reference-bundle-v1-20260923', 'rod-height-bundle-v1-20260923']):
        for cid in ['r01', 'r02', 'r03']:
            rows = [r for r in data['rows'] if r['parent'] == parent and r['case_id'] == cid and r['method'] == 'baseline' and r['state'] == 'accepted']
            if not rows:
                continue
            for control, marker, size in [(False, 'o', 36), (True, '*', 180)]:
                selected = [r for r in rows if (r['condition'] == 'control') == control]
                ax.scatter([r['all_view_residual']['p95_px'] for r in selected],
                    [1000 * r['metrics']['truth_to_prediction']['distance_p95'] for r in selected],
                    c=palette[cid], marker=marker, s=size, edgecolor='white', linewidth=.6,
                    label=names[cid] if not control else None, zorder=3 if control else 2)
        ax.set_title('Same-height path' if 'reference' in parent else 'Varied-height path')
        ax.set_xlabel('All-view held-out reprojection p95 (px)')
        ax.set_ylabel('Finite target-to-curve p95 (mm)')
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
        ax.set_ylim(bottom=0)
    fig.suptitle('Small pixel residuals do not certify the rod geometry', fontsize=13)
    fig.text(.5, -.04, 'Six fixed-lens cases, ordinary rod method; stars = full, circles = leave-groups. Correlated folds.\nAccepted curves only. Zoom negative remains in the tables. No fold is promoted.',
             ha='center', fontsize=9)
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=170, bbox_inches='tight')
    plt.close(fig)
    for case in cases:
        row = case['methods']['baseline']
        print(case['parent'], case['case_id'], row['fold_states'], row['fold_recovery'], row['fold_precision'], row['fold_p95_m'])
    print('REPORT', output, figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--figure', type=Path, default=FIGURE)
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    report(args.source, args.output, args.figure)
