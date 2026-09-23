"""Audit the actual paired acquisition change before comparing all frozen outcomes."""
from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image
from prepare_rod_reference import checked_scene
from run_rod_reference import ROOT, digest, read_json, write_json

RUN_ID = 'rod-height-bundle-v1-20260923'


def rgb_digest(path):
    with Image.open(path) as image:
        pixels = np.asarray(image.convert('RGB'))
    return hashlib.sha256(str(pixels.shape).encode('ascii') + pixels.tobytes()).hexdigest()


def actual_pair_audit():
    config = read_json(ROOT / 'configs/rod_height_v1.json')
    current = checked_scene(config['run_id'])
    previous = checked_scene(config['paired_scene_run_id'])
    records = []
    for scene in (current, previous):
        run, _, truth, frozen, _ = scene
        assert digest(run / 'protocol.json') == frozen['protocol_sha256']
        assert digest(truth / 'artifact_hashes.json') == frozen['truth_artifacts_sha256']
        for name, sha in read_json(truth / 'artifact_hashes.json').items():
            assert digest(truth / name) == sha
    new_truth = read_json(current[2] / 'manifest.json')
    old_truth = read_json(previous[2] / 'manifest.json')
    for cid in [c['case_id'] for c in config['cases']]:
        before = next(c for c in old_truth['cases'] if c['case_id'] == cid)
        after = next(c for c in new_truth['cases'] if c['case_id'] == cid)
        np.testing.assert_equal(before['declared'], after['declared'])
        assert digest(previous[2] / before['mesh_path']) == digest(current[2] / after['mesh_path'])
        old_frames = next(c for c in previous[4]['cases'] if c['case_id'] == cid)['frames']
        new_frames = next(c for c in current[4]['cases'] if c['case_id'] == cid)['frames']
        assert [f['view_id'] for f in old_frames] == [f['view_id'] for f in new_frames]
        first_old, first_new = ROOT / old_frames[0]['rgb'], ROOT / new_frames[0]['rgb']
        assert rgb_digest(first_old) == rgb_digest(first_new)
        with Image.open(first_old) as a, Image.open(first_new) as b:
            changed_metadata = [key for key in a.info.keys() | b.info.keys() if a.info.get(key) != b.info.get(key)]
        # PNG stamps contain wall-clock Date/RenderTime; file bytes differ even
        # when every decoded RGB value agrees. Keep both identities explicit.
        assert set(changed_metadata) <= {'Date', 'RenderTime'}
        old_centers, new_centers = [], []
        for a, b, height in zip(before['cameras'], after['cameras'], config['generator']['camera_heights_m']):
            np.testing.assert_array_equal(a['K_index'], b['K_index'])
            ca = np.linalg.inv(a['world_to_camera_cv'])[:3, 3]
            cb = np.linalg.inv(b['world_to_camera_cv'])[:3, 3]
            np.testing.assert_allclose(ca[:2], cb[:2], atol=2e-6, rtol=0)
            assert abs(cb[2] - height) < 2e-6
            e = np.asarray(b['world_to_camera_cv'])
            np.testing.assert_allclose(e[2, :3], -cb / np.linalg.norm(cb), atol=2e-6)
            old_centers.append(ca)
            new_centers.append(cb)
        # An independent byte check of every PNG also binds this audit to the
        # actual input files, rather than trusting names in two manifests.
        for f in old_frames + new_frames:
            assert digest(ROOT / f['rgb']) == f['rgb_sha256']
        records.append(dict(case_id=cid, mesh_unchanged=True, lens_and_xy_unchanged=True,
            first_rgb_exact=True, first_png_bytes_equal=old_frames[0]['rgb_sha256'] == new_frames[0]['rgb_sha256'],
            first_rgb_sha256=rgb_digest(first_new), first_changed_metadata=changed_metadata,
            old_centers=old_centers, new_centers=new_centers,
            changed_rgb_views=sum(rgb_digest(ROOT / a['rgb']) != rgb_digest(ROOT / b['rgb']) for a, b in zip(old_frames, new_frames))))
    return records, current, previous


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    audit, current, previous = actual_pair_audit()
    old_path = ROOT / 'docs/experiments/results/2026-09-23-rod-reference.json'
    new_path = ROOT / 'docs/experiments/results/2026-09-23-rod-height.json'
    old, new = read_json(old_path), read_json(new_path)
    assert old['evaluation_source_sha256'] == new['evaluation_source_sha256'] == digest(ROOT / 'scripts/evaluate_rod_reference.py')
    pairs = []
    for row in new['rods']:
        old_row = next(r for r in old['rods'] if (r['case_id'], r['camera_variant'], r['method']) == (row['case_id'], row['camera_variant'], row['method']))
        pairs.append(dict(case_id=row['case_id'], variant=row['camera_variant'], method=row['method'], before=old_row, after=row))
    camera_pairs = []
    for row in new['cameras']:
        before = next(r for r in old['cameras'] if (r['case_id'], r['variant']) == (row['case_id'], row['variant']))
        camera_pairs.append(dict(case_id=row['case_id'], variant=row['variant'],
            before={key: before[key] for key in ('heldout', 'camera', 'decision')},
            after={key: row[key] for key in ('heldout', 'camera', 'decision')}))
    names = ['r01', 'r02', 'r03']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), layout='constrained')
    x = np.arange(3)
    for side, label, offset, color in (('before', 'Same height', -.18, '#888888'), ('after', 'Varied height', .18, '#17699b')):
        cameras = [next(r for r in camera_pairs if r['case_id'] == cid and r['variant'] == 'common_K_focal_trimmed')[side] for cid in names]
        rods = [next(r for r in pairs if r['case_id'] == cid and r['variant'] == 'common_K_focal_trimmed' and r['method'] == 'baseline')[side] for cid in names]
        axes[0].bar(x + offset, [r['camera']['camera_rmse_m'] * 1000 for r in cameras], .35, label=label, color=color)
        for ax, field in ((axes[1], 'recovery_fraction'), (axes[2], 'precision_fraction')):
            ax.bar(x + offset, [100 * (r['metrics'][field] or 0) for r in rods], .35, color=color)
            for j, r in enumerate(rods):
                if r['state'] != 'accepted':
                    ax.plot(j + offset, 1, 'x', color=color)
    axes[0].set_ylabel('Camera-center RMSE (mm)')
    axes[1].set(ylabel='Rod recovered length (%)', ylim=(-3, 106))
    axes[2].set(ylabel='Rod precision (%)', ylim=(-3, 106))
    for ax in axes:
        ax.set(xticks=x, xticklabels=['Complete', 'Gap', 'Thin'])
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    axes[0].legend()
    fig.suptitle('Camera-height intervention | frozen two-stage calibration | ordinary rod output')
    fig.supxlabel('Same 3 known rod layouts; 25 mm tolerance; no independent-object or dense-patch efficacy claim.', fontsize=9)
    figure = ROOT / 'docs/experiments/figures/2026-09-23-rod-height.png'
    fig.savefig(figure, dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), layout='constrained')
    for i, (scene, title) in enumerate(((previous, 'Same height'), (current, 'Varied height'))):
        for ax, cid in zip(axes[i], names):
            frame = next(c for c in scene[4]['cases'] if c['case_id'] == cid)['frames'][3]
            ax.imshow(Image.open(ROOT / frame['rgb']))
            ax.set_title(title + ' / ' + cid)
            ax.axis('off')
    overview = ROOT / 'docs/experiments/figures/2026-09-23-rod-height-rgb.png'
    fig.savefig(overview, dpi=140)
    plt.close(fig)
    write_json(ROOT / 'docs/experiments/results/2026-09-23-rod-height-paired.json', dict(state='complete', run_id=RUN_ID,
        before_sha256=digest(old_path), after_sha256=digest(new_path), acquisition_audit=audit,
        camera_pairs=camera_pairs, rod_pairs=pairs, report_source_sha256=digest(__file__),
        figures={p.name: digest(p) for p in (figure, overview)}))
    print('HEIGHT_PAIRED_AUDIT', len(pairs), 'rod rows; unchanged meshes/K/XY and first RGB', flush=True)


if __name__ == '__main__':
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    report()
