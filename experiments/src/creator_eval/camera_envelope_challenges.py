"""New analytic pixel challenges with strict input/truth separation.

This creates no RGB images and runs no estimator. Known track and endpoint
correspondence is an explicit simulation assumption, not a detector result.
Initial cameras are declared synthetic perturbations, never DA3 outputs.
"""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np

INPUT_CASE_KEYS = {'case_id', 'size_wh', 'view_ids', 'training', 'validation',
                   'initial_intrinsics', 'initial_extrinsics', 'rod_tracks', 'rod_support_sha256'}


def rod_support_digest(rod_tracks):
    encoded = json.dumps(rod_tracks, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _rotation(vector):
    angle = float(np.linalg.norm(vector))
    if angle < 1e-15:
        return np.eye(3)
    x, y, z = np.asarray(vector, float) / angle
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + np.sin(angle) * skew + (1. - np.cos(angle)) * (skew @ skew)


def _noncollinearity(centers):
    singular = np.linalg.svd(centers - centers.mean(axis=0), compute_uv=False)
    return float(singular[1] / singular[0]) if singular[0] > 1e-12 else 0.


def _cameras(generation, specification, rng):
    size = generation['size_wh']
    intrinsic = np.array([[generation['focal_px'], 0., (size[0] - 1.) / 2],
                          [0., generation['focal_px'], (size[1] - 1.) / 2], [0., 0., 1.]])
    layout = np.asarray(generation['camera_layout_unit'], float)
    if layout.shape != (5, 3) or not np.isfinite(layout).all():
        raise ValueError('Five finite camera centers required')
    baseline = float(np.linalg.norm(layout[-1] - layout[0]))
    if baseline <= 1e-8:
        raise ValueError('Camera layout baseline is degenerate')
    centers = layout * float(specification['camera_baseline_m']) / baseline
    if _noncollinearity(centers) < .01:
        raise ValueError('True camera layout must allow camera-only alignment')
    extrinsics = []
    target = np.asarray(generation['look_at_world'], float)
    for center in centers:
        forward = target - center
        forward /= np.linalg.norm(forward)
        right = np.cross([0., 1., 0.], forward)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rotation = np.stack([right, down, forward])
        extrinsics.append(np.c_[rotation, -rotation @ center])
    extrinsics = np.asarray(extrinsics)
    intrinsics = np.repeat(intrinsic[None], len(centers), axis=0)
    parameters = generation['initialization']
    if parameters['position_std_m'] <= 0 or parameters['rotation_std_degrees'] <= 0 or parameters['focal_relative_bias'] <= 0:
        raise ValueError('This protocol requires declared nonzero perturbations, not exact-camera initialization')
    location_noise = rng.normal(0., parameters['position_std_m'], centers.shape)
    rotation_noise = rng.normal(0., np.radians(parameters['rotation_std_degrees']), centers.shape)
    focal_noise = rng.normal(0., parameters['focal_log_noise_std'], len(centers))
    estimated_centers = centers + location_noise
    if _noncollinearity(estimated_centers) < .001:
        raise ValueError('Perturbed initial centers are degenerate; do not silently choose an alignment rotation')
    estimated_e, estimated_k = [], intrinsics.copy()
    for view, (center, original) in enumerate(zip(estimated_centers, extrinsics)):
        rotation = _rotation(rotation_noise[view]) @ original[:, :3]
        estimated_e.append(np.c_[rotation, -rotation @ center])
        factor = (1. + parameters['focal_relative_bias']) * np.exp(focal_noise[view])
        estimated_k[view, 0, 0] *= factor
        estimated_k[view, 1, 1] *= factor
    details = dict(position_noise_m=location_noise.tolist(), rotation_noise_rad=rotation_noise.tolist(),
                   focal_log_noise=focal_noise.tolist(), true_center_noncollinearity=_noncollinearity(centers),
                   initial_center_noncollinearity=_noncollinearity(estimated_centers),
                   source='synthetic_latent_camera_plus_preregistered_perturbation_not_DA3')
    return intrinsics, extrinsics, estimated_k, np.asarray(estimated_e), details


def _project(points, intrinsics, extrinsics):
    points = np.asarray(points, float).reshape(-1, 3)
    camera = np.einsum('vij,nj->vni', extrinsics[:, :, :3], points) + extrinsics[:, None, :, 3]
    pixels = np.einsum('vij,vnj->vni', intrinsics, camera)
    if not np.isfinite(camera).all() or np.any(camera[..., 2] <= 0):
        raise ValueError('Generated point is behind a camera')
    return pixels[..., :2] / pixels[..., 2, None]


def _background(split, count, generation, specification, intrinsics, extrinsics, rng, bias):
    if count == 0:
        return [], []
    y_range = generation['training_sampling_y_px'] if split == 'training' else generation['validation_sampling_y_px']
    inverse = np.linalg.inv(intrinsics[0])
    rotation = extrinsics[0, :, :3]
    center = -rotation.T @ extrinsics[0, :, 3]
    size, margin = np.asarray(generation['size_wh']), generation['accepted_image_margin_px']
    observed, world = [], []
    for _ in range(generation['maximum_proposals_per_split']):
        u = rng.uniform(*generation['sampling_x_px'])
        if abs(u - intrinsics[0, 0, 2]) < generation['central_exclusion_half_width_px']:
            continue
        v = rng.uniform(*y_range)
        depth = rng.uniform(*generation['volumetric_depth_world_z_m']) if specification['background'] == 'volumetric' else {'plane_8m': 8., 'plane_24m': 24.}[specification['background']]
        direction = rotation.T @ inverse @ np.array([u, v, 1.])
        point = center + direction * ((depth - center[2]) / direction[2])
        xy = _project(point[None], intrinsics, extrinsics)[:, 0]
        xy += rng.normal(0., specification['pixel_noise_std'], xy.shape) + bias
        if np.any(xy < margin) or np.any(xy > size - 1 - margin):
            continue
        in_band = np.all(xy[:, 1] >= 192.) if split == 'training' else np.all(xy[:, 1] < 128.)
        if not in_band:
            continue
        # 只按实际带噪像素所在区域拆分。这里不能拿真值误差筛一批漂亮轨迹。
        observed.append([dict(view=int(i), xy=value.tolist()) for i, value in enumerate(xy)])
        world.append(point.tolist())
        if len(observed) == count:
            return observed, world
    raise ValueError('Declared sampling budget cannot supply the frozen split')


def _rod_data(generation, specification, intrinsics, extrinsics, rng, bias):
    low, high = generation['rod_y_endpoints_m']
    depth = float(specification['rod_depth_m'])
    mode = specification['measurement_mode']
    spans = [[low, high]]
    if mode == 'gap':
        a, b = generation['gap_y_m']
        spans = [[low, a], [b, high]]
    truth = np.asarray([[[0., a, depth], [0., b, depth]] for a, b in spans])
    measured = truth.copy()
    if mode == 'coherent_lateral':
        measured[:, :, 0] += generation['coherent_lateral_offset_m']
    if mode == 'coherent_depth':
        measured[:, :, 2] *= generation['coherent_depth_factor']
    records = []
    for segment in measured:
        projected = _project(segment, intrinsics, extrinsics)
        projected += rng.normal(0., specification['pixel_noise_std'], projected.shape) + bias[:, None, :]
        if mode == 'endpoint_swap':
            view = generation['swap_view_index']
            projected[view] = projected[view, ::-1].copy()
        visible_views = list(range(len(intrinsics)))
        if mode == 'single_view':
            visible_views = [generation['single_observed_view_index']]
        elif mode == 'no_observations':
            visible_views = []
        endpoints = [dict(track_id='p' + str(int(rng.integers(10**8, 10**9))),
                          observations=[dict(view=view, xy=projected[view, endpoint].tolist()) for view in visible_views])
                     for endpoint in range(2)]
        records.append(dict(segment_id='s' + str(int(rng.integers(10**8, 10**9))), endpoint_tracks=endpoints))
    ids = [r['segment_id'] for r in records] + [e['track_id'] for r in records for e in r['endpoint_tracks']]
    if len(ids) != len(set(ids)):
        raise ValueError('Anonymous rod ids collided; preserve the failed generation')
    return records, truth.tolist(), measured.tolist()


def generate_challenges(protocol):
    """Return separate `(inputs, truth)` dictionaries; caller freezes/writes them.

    Generation configuration, seeds, family labels, latent points and true cameras
    only enter `truth`/the protocol. The inference dictionary has a fixed whitelist.
    Rod segment order equals truth `rod_segments` order, including the gap case.
    """
    if protocol['schema_version'] != 'analytic-camera-envelope-protocol-v1' or len(protocol['cases']) != protocol['expected_case_count']:
        raise ValueError('Declared analytic protocol/case count mismatch')
    generation = copy.deepcopy(protocol['generation'])
    if generation['size_wh'] != [640, 480] or len({c['family'] for c in protocol['cases']}) != len(protocol['cases']):
        raise ValueError('Fixed image convention and unique declared families required')
    anonymous_order = np.random.default_rng(protocol['anonymization_seed']).permutation(len(protocol['cases']))
    inputs, truths = [], []
    for ordinal, specification in enumerate(protocol['cases']):
        specification = copy.deepcopy(specification)
        if specification['camera_baseline_m'] <= 0 or specification['pixel_noise_std'] < 0 or specification['rod_depth_m'] <= 0:
            raise ValueError('Positive geometry scales and nonnegative noise required')
        rng_init = np.random.default_rng(specification['scene_seed'] + 1000000)
        rng_pixels = np.random.default_rng(specification['scene_seed'] + 2000000)
        rng_rod = np.random.default_rng(specification['scene_seed'] + 3000000)
        rng_ids = np.random.default_rng(specification['scene_seed'] + 4000000)
        k, e, initial_k, initial_e, initialization = _cameras(generation, specification, rng_init)
        view_ids = [f'v{view:02d}' for view in range(len(k))]
        case_id = f'c{int(anonymous_order[ordinal]) + 1:03d}'
        bias = np.asarray(generation['frame_bias_uv_px'], float) if specification['measurement_mode'] == 'frame_bias' else np.zeros((len(k), 2))
        empty = specification['measurement_mode'] == 'no_observations'
        train, train_world = _background('training', 0 if empty else generation['training_count'], generation, specification, k, e, rng_pixels, bias)
        val, val_world = _background('validation', 0 if empty else generation['validation_count'], generation, specification, k, e, rng_pixels, bias)
        ids = rng_ids.permutation(np.arange(10000, 10000 + len(train) + len(val))).tolist()
        training = [dict(track_id=int(tid), observations=observations) for tid, observations in zip(ids[:len(train)], train)]
        validation = [dict(track_id=int(tid), observations=observations) for tid, observations in zip(ids[len(train):], val)]
        rod_tracks, rod_segments, measurement_segments = _rod_data(generation, specification, k, e, rng_rod, bias)
        item = dict(case_id=case_id, size_wh=copy.deepcopy(generation['size_wh']), view_ids=view_ids,
                    training=sorted(training, key=lambda t: t['track_id']), validation=sorted(validation, key=lambda t: t['track_id']),
                    initial_intrinsics=initial_k.tolist(), initial_extrinsics=initial_e.tolist(),
                    rod_tracks=rod_tracks, rod_support_sha256=rod_support_digest(rod_tracks))
        if set(item) != INPUT_CASE_KEYS:
            raise ValueError('Inference schema gained an undeclared field')
        inputs.append(item)
        cameras = []
        for view in range(len(k)):
            matrix = np.eye(4)
            matrix[:3] = e[view]
            cameras.append(dict(view_id=view_ids[view], K_index=k[view].tolist(), world_to_camera_cv=matrix.tolist(),
                                size_wh=copy.deepcopy(generation['size_wh'])))
        truth = dict(case_id=case_id, family=specification['family'], intended_role=specification['intended_role'],
                     expected_limit=specification['expected_limit'], cameras=cameras, rod_segments=rod_segments,
                     generation_parameters=specification, initialization=initialization,
                     measurement_rod_segments=measurement_segments, shared_frame_pixel_bias=bias.tolist(),
                     background_points=[dict(track_id=int(tid), world_xyz=point) for tid, point in zip(ids, train_world + val_world)],
                     known_cross_view_endpoint_correspondence='Analytic assumption; fixed wrong associations in declared counterexamples, never an RGB matching result')
        truths.append(truth)
    inputs.sort(key=lambda case: case['case_id'])
    truths.sort(key=lambda case: case['case_id'])
    common = dict(schema_version='analytic-camera-envelope-data-v1', dataset_id=protocol['dataset_id'])
    return ({**common, 'scope': 'Anonymous analytic pixel measurements; no rendered RGB or DA3 prediction claim',
             'initialization': 'synthetic_latent_camera_plus_preregistered_perturbation_not_DA3; exact true cameras are evaluation-only',
             'cases': inputs},
            {**common, 'scope': 'Evaluation-only latent geometry, labels and generation details; never estimator inputs', 'cases': truths})
