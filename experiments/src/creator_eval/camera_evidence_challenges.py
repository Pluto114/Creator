"""Paired analytic evidence challenges; synthetic pixels are never RGB claims.

Four rod measurement modes share one background/camera draw. Target claims are
a second simulated measurement channel, not human annotations or recognition.
This generator performs no fitting, candidate selection, or physical scoring.
"""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np

from creator_eval.camera_envelope_challenges import (
    _project,
    _rod_data,
    generate_challenges,
    rod_support_digest,
)

MODES = ("ordinary", "endpoint_swap", "coherent_lateral", "coherent_depth")
GROUP_KEYS = {"camera_group_id", "size_wh", "view_ids", "training", "validation",
              "initial_intrinsics", "initial_extrinsics", "frames"}
CASE_KEYS = {"case_id", "camera_group_id", "rod_tracks", "rod_support_sha256", "evidence_sets"}
ANCHOR_KEYS = {"view_id", "xy", "uncertainty_xy_px", "source_sha256"}
EVIDENCE_ROLES = ("target", "candidate_consistent_target", "single_view_target")


def canonical_digest(value):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _anchors(segment, k, e, view_ids, config, noise):
    segment = np.asarray(segment, float)
    points = segment[0] + np.asarray(config["fractions"])[:, None] * (segment[1] - segment[0])
    pixels = _project(points, k, e)
    return [dict(view_id=view_ids[view], xy=(pixels[view, index] + noise[index]).tolist(),
                 uncertainty_xy_px=copy.deepcopy(config["uncertainty_xy_px"]))
            for index, view in enumerate(config["views"])]


def _frame_banks(group, cases):
    """Bind the pixels, without hashing a hash field back into its own input."""
    frames = []
    for view, view_id in enumerate(group["view_ids"]):
        bank = dict(view_id=view_id, training=[], validation=[], rod_observations=[], anchor_observations=[])
        for label in ("training", "validation"):
            for track in group[label]:
                for observation in track["observations"]:
                    if observation["view"] == view:
                        bank[label].append(dict(track_id=track["track_id"], xy=copy.deepcopy(observation["xy"])))
        for case in sorted(cases, key=lambda c: c["case_id"]):
            for segment in case["rod_tracks"]:
                for endpoint in segment["endpoint_tracks"]:
                    for observation in endpoint["observations"]:
                        if observation["view"] == view:
                            bank["rod_observations"].append(dict(
                                case_id=case["case_id"], segment_id=segment["segment_id"],
                                endpoint_track_id=endpoint["track_id"], xy=copy.deepcopy(observation["xy"])))
            for evidence in sorted(case["evidence_sets"], key=lambda item: item["evidence_id"]):
                for anchor in evidence["anchors"]:
                    if anchor["view_id"] == view_id:
                        bank["anchor_observations"].append(dict(
                            case_id=case["case_id"], evidence_id=evidence["evidence_id"],
                            xy=copy.deepcopy(anchor["xy"]),
                            uncertainty_xy_px=copy.deepcopy(anchor["uncertainty_xy_px"])))
        frames.append(dict(view_id=view_id, size_wh=copy.deepcopy(group["size_wh"]),
                           source_kind="synthetic_pixel_measurements",
                           source_sha256=canonical_digest(bank), measurement_bank=bank))
    return frames


def generate_paired_challenges(protocol):
    """Return separate inputs/truth; the caller freezes sources before writing.

    Input IDs are anonymous but public pairing is intentional. No input contains
    seeds, physical target labels, latent points or an exact true camera. The
    old generator dependency must be included in the caller's source freeze.
    """
    seeds = protocol["seeds"]
    if protocol["schema_version"] != "paired-camera-evidence-protocol-v1":
        raise ValueError("Unexpected paired evidence protocol")
    if len(seeds) != protocol["expected_seed_count"] or len(set(seeds)) != len(seeds):
        raise ValueError("Declared unique seed count mismatch")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("Nonnegative integer seeds required")
    if tuple(protocol["modes"]) != MODES:
        raise ValueError("Exactly the four preregistered paired modes required")
    generation, scene, anchor_config = protocol["generation"], protocol["scene"], protocol["anchors"]
    if anchor_config["views"] != [0, 4] or anchor_config["fractions"] != [.35, .65]:
        raise ValueError("Target claim views/fractions changed")
    if anchor_config["noise_std_px"] < 0 or anchor_config["uncertainty_xy_px"] != [.5, .5]:
        raise ValueError("Invalid target claim noise or fixed uncertainty")
    if (scene["background"] != "volumetric" or scene["rod_depth_m"] != 4.
            or scene["pixel_noise_std"] < 0 or scene["camera_baseline_m"] <= 0):
        raise ValueError("Fixed volumetric near-rod scenario required")
    rng_names = np.random.default_rng(protocol["anonymization_seed"])
    group_order = rng_names.permutation(len(seeds))
    case_order = rng_names.permutation(len(seeds) * len(MODES))
    groups, cases, hidden_groups, hidden_cases = [], [], [], []
    for ordinal, seed in enumerate(seeds):
        group_id = f"g{int(group_order[ordinal]) + 1:03d}"
        specification = dict(scene, scene_seed=seed, family="paired_base",
                             measurement_mode="ordinary", intended_role="paired_component_challenge",
                             expected_limit="Synthetic perturbed initialization and supplied endpoint identities")
        # Generate the background once. Rebuilding it in each mode is too easy
        # to get subtly wrong when a future mode consumes one extra RNG draw.
        base_protocol = dict(schema_version="analytic-camera-envelope-protocol-v1",
                             dataset_id=protocol["dataset_id"], expected_case_count=1,
                             anonymization_seed=0, generation=copy.deepcopy(generation), cases=[specification])
        base_inputs, base_truth = generate_challenges(base_protocol)
        base, hidden = base_inputs["cases"][0], base_truth["cases"][0]
        group = {key: copy.deepcopy(base[key]) for key in
                 ("size_wh", "view_ids", "training", "validation", "initial_intrinsics", "initial_extrinsics")}
        group["camera_group_id"] = group_id
        k = np.asarray([camera["K_index"] for camera in hidden["cameras"]])
        e = np.asarray([camera["world_to_camera_cv"][:3] for camera in hidden["cameras"]])
        anchor_noise = np.random.default_rng(seed + 5000000).normal(
            0., anchor_config["noise_std_px"], (len(anchor_config["views"]), 2))
        true_anchors = _anchors(hidden["rod_segments"][0], k, e, group["view_ids"], anchor_config, anchor_noise)
        siblings = []
        for mode_index, mode in enumerate(MODES):
            case_id = f"c{int(case_order[ordinal * len(MODES) + mode_index]) + 1:03d}"
            mode_spec = dict(specification, measurement_mode=mode)
            rod_tracks, rod_segments, measured = _rod_data(
                generation, mode_spec, k, e, np.random.default_rng(seed + 3000000), np.zeros((len(k), 2)))
            if mode == "ordinary" and rod_tracks != base["rod_tracks"]:
                raise ValueError("Paired control detached from the base rod draw")
            wrong_anchors = _anchors(measured[0], k, e, group["view_ids"], anchor_config, anchor_noise)
            # Swapping endpoint identity in one view does not create a different
            # physical rod. Its target-claim control must therefore stay equal.
            if mode in ("ordinary", "endpoint_swap") and wrong_anchors != true_anchors:
                raise ValueError("Equivalent target-claim controls diverged")
            claims = [true_anchors, wrong_anchors, true_anchors[:1]]
            evidence_order = rng_names.permutation(len(EVIDENCE_ROLES))
            evidence_sets, hidden_evidence = [], []
            for index, (role, anchors) in enumerate(zip(EVIDENCE_ROLES, claims)):
                evidence_id = f"e{int(evidence_order[index]) + 1:02d}"
                evidence_sets.append(dict(evidence_id=evidence_id, source_kind="synthetic_2d_claim",
                                          anchors=copy.deepcopy(anchors)))
                hidden_evidence.append(dict(evidence_id=evidence_id, role=role,
                    physical_source="measurement_rod" if role == "candidate_consistent_target" else "requested_rod",
                    equivalent_to_target=mode in ("ordinary", "endpoint_swap") and role == "candidate_consistent_target",
                    point_fractions=copy.deepcopy(anchor_config["fractions"][:len(anchors)]),
                    noise_uv_px=anchor_noise[:len(anchors)].tolist()))
            case = dict(case_id=case_id, camera_group_id=group_id, rod_tracks=rod_tracks,
                        rod_support_sha256=rod_support_digest(rod_tracks),
                        evidence_sets=sorted(evidence_sets, key=lambda item: item["evidence_id"]))
            if set(case) != CASE_KEYS:
                raise ValueError("Paired inference case schema changed")
            siblings.append(case)
            hidden_cases.append(dict(case_id=case_id, camera_group_id=group_id, mode=mode,
                rod_segments=rod_segments, measurement_rod_segments=measured,
                evidence_labels=sorted(hidden_evidence, key=lambda item: item["evidence_id"]),
                boundary="Coherent wrong rods and equally wrong target claims can agree without being correct"
                         if mode.startswith("coherent_") else "Endpoint identity stress" if mode == "endpoint_swap" else "Paired clean control"))
        group["frames"] = _frame_banks(group, siblings)
        by_view = {frame["view_id"]: frame for frame in group["frames"]}
        for case in siblings:
            for evidence in case["evidence_sets"]:
                for anchor in evidence["anchors"]:
                    anchor["source_sha256"] = by_view[anchor["view_id"]]["source_sha256"]
                    if set(anchor) != ANCHOR_KEYS:
                        raise ValueError("Paired anchor schema changed")
                evidence["evidence_sha256"] = canonical_digest(
                    {key: evidence[key] for key in ("source_kind", "anchors")})
        if set(group) != GROUP_KEYS:
            raise ValueError("Paired inference camera schema changed")
        groups.append(group)
        cases.extend(siblings)
        hidden_groups.append(dict(camera_group_id=group_id, seed=seed, cameras=hidden["cameras"],
            initialization=hidden["initialization"], background_points=hidden["background_points"],
            generation_parameters=specification, endpoint_noise_stream_seed=seed + 3000000,
            anchor_noise_stream_seed=seed + 5000000, anchor_noise_uv_px=anchor_noise.tolist()))
    common = dict(schema_version="paired-camera-evidence-data-v1", dataset_id=protocol["dataset_id"])
    inputs = dict(**common, scope="Paired synthetic pixel measurements and target claims; no RGB, human annotation, or automatic target-recognition claim",
                  initialization="synthetic_latent_camera_plus_preregistered_perturbation_not_DA3",
                  camera_groups=sorted(groups, key=lambda item: item["camera_group_id"]),
                  cases=sorted(cases, key=lambda item: item["case_id"]))
    truth = dict(**common, scope="Evaluation-only true geometry, seeds, modes and claim labels",
                 camera_groups=sorted(hidden_groups, key=lambda item: item["camera_group_id"]),
                 cases=sorted(hidden_cases, key=lambda item: item["case_id"]))
    # This also rejects accidental NaN/Inf before the writer can freeze a run.
    json.dumps(inputs, allow_nan=False)
    json.dumps(truth, allow_nan=False)
    return inputs, truth
