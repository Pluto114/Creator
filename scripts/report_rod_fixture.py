"""Draw the fixed middle view and native finite predictions, with source receipts.

This is a presentation step after inference. It reads no GT and never changes
camera estimates, pixel evidence, method decisions, or the selected view.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / '.runtime/experiments/rod-fixture-finite-v1-20260927'
FIGURE = ROOT / 'docs/experiments/figures/2026-09-27-fixture-finite-rgb.png'
RECEIPT = ROOT / 'docs/experiments/results/2026-09-27-fixture-figure.json'
COLORS = {'baseline': '#36d5ff', 'cylinder_support': '#ffb84d'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    if FIGURE.exists() or RECEIPT.exists():
        raise FileExistsError('Preserve the published figure and its source receipt')
    index = read(RUN / 'inference.json')
    assert index['state'] == 'complete' and index['gt_read'] is False
    records = index['records']
    assert [r['case_id'] for r in records] == ['r01', 'r02', 'r03']
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12})
    figure, axes = plt.subplots(3, 2, figsize=(12, 12), gridspec_kw={'width_ratios': [1.7, 1.]})
    receipts = []
    for row_index, receipt in enumerate(records):
        path = RUN / receipt['path']
        assert digest(path) == receipt['sha256']
        record = read(path)
        frame = record['frames'][2]
        camera = record['cameras'][2]
        assert frame['view_id'] == camera['view_id'] == 'view_02'
        rgb_path = ROOT / frame['rgb']
        assert digest(rgb_path) == frame['rgb_sha256']
        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert('RGB'))
        left, right = axes[row_index]
        left.imshow(rgb)
        left.set_title(f"{receipt['case_id']} / view_02 — original RGB")
        left.set_xlabel('Native x pixel')
        left.set_ylabel('Native y pixel')
        left.set_xlim(-.5, 639.5)
        left.set_ylim(479.5, -.5)
        right.imshow(rgb, aspect='auto')
        right.set_xlim(295, 344)
        right.set_ylim(430, 50)
        right.set_title('Finite predictions in the search strip')
        right.set_xlabel('Native x pixel (horizontal magnification)')
        right.set_ylabel('Native y pixel')
        notes = []
        for method in record['result']['methods']:
            name = method['method']
            segments = np.asarray(method['segments'], float).reshape(-1, 2, 3)
            notes.append(f"{name}: {method['state']} / {len(segments)} segment(s)")
            if len(segments):
                k = np.asarray(camera['K_index'])
                e = np.asarray(camera['world_to_camera_cv'])[:3]
                for i, segment in enumerate(segments):
                    transformed = np.c_[segment, np.ones(2)] @ e.T
                    assert np.all(transformed[:, 2] > 0)
                    projected = transformed @ k.T
                    xy = projected[:, :2] / projected[:, 2, None]
                    # Draw both methods at their actual coordinates. Offset lines
                    # would look cleaner but would quietly invent a discrepancy.
                    right.plot(xy[:, 0], xy[:, 1], color=COLORS[name],
                        linewidth=2.2 if name == 'baseline' else 1.4,
                        linestyle='-' if name == 'baseline' else '--',
                        marker='o' if name == 'baseline' else 'x', markersize=4,
                        label=name if i == 0 else None)
            else:
                right.plot([], [], color=COLORS[name], label=name + ' (no output)')
        left.text(0., -.20, '\n'.join(notes), transform=left.transAxes, va='top', fontsize=9)
        receipts.append(dict(case_id=receipt['case_id'], record_sha256=receipt['sha256'],
            view_id=frame['view_id'], rgb_sha256=frame['rgb_sha256'],
            camera_state=camera['state'], methods=[dict(method=m['method'], state=m['state'],
                segment_count=m['segment_count']) for m in record['result']['methods']]))
    figure.legend(*axes[0, 1].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.5, .048), ncol=2, fontsize=10)
    figure.suptitle('Known metric fixture / unchanged finite RGB rules', fontsize=17, y=.995)
    figure.text(.5, .006,
        'Fixed middle view for all three cases. Insets stretch x for inspection; pixel coordinates are unchanged.\n'
        'Projection uses estimated fixture cameras. Visual agreement is not a physical-accuracy certificate.',
        ha='center', fontsize=10)
    figure.subplots_adjust(left=.07, right=.98, top=.955, bottom=.16, hspace=.52, wspace=.25)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE, dpi=150)
    plt.close(figure)
    result = dict(state='complete', run_id=index['run_id'],
        script_sha256=digest(Path(__file__)), inference_sha256=digest(RUN / 'inference.json'),
        figure=FIGURE.relative_to(ROOT).as_posix(), figure_sha256=digest(FIGURE),
        selected_view_policy='Fixed ordered third frame / view_02 for every case',
        crop_pixel_bounds=dict(x=[295, 344], y=[50, 430]), ground_truth_read=False, records=receipts)
    with RECEIPT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('FIXTURE_FIGURE', FIGURE.relative_to(ROOT).as_posix(), flush=True)


if __name__ == '__main__':
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    main()
