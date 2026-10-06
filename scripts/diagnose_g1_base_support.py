"""Audit existing base support without reading truth or changing any reader policy."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import run_fixture_consensus_readout_v2 as fixture
import scipy
from creator_eval.rgb_candidate_readout import POLICY, support_views, support_votes
from creator_recon.domain.point_patch import file_hash, load_snapshot
from scipy.spatial import cKDTree

ROOT = fixture.ROOT
PUBLIC = ROOT / "docs/experiments/results/2026-10-06-g1-base-support.json"
CHUNK_POINTS = 32768
QUANTILES = (0., .25, .5, .75, .95, 1.)


def block_truth(event, arguments):
    """Allow only the exact new report's exclusive write, never score reads."""
    if event == "open" and arguments and isinstance(arguments[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(arguments[0])).resolve()
        mode = arguments[1] if len(arguments) > 1 else None
        flags = arguments[2] if len(arguments) > 2 else 0
        if (path == PUBLIC.resolve() and mode == "x" and isinstance(flags, int)
                and flags & os.O_EXCL and flags & os.O_WRONLY and not flags & os.O_RDWR):
            return
    fixture.block_truth(event, arguments)


def verify_receipts(hashes):
    for name, expected in hashes.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT.resolve()) or file_hash(path) != expected:
            raise ValueError("Changed or out-of-root normal receipt: " + name)


def bind(path, hashes, expected=None):
    path = Path(path).resolve()
    name, actual = path.relative_to(ROOT.resolve()).as_posix(), file_hash(path)
    if (expected is not None and actual != expected) or (name in hashes and hashes[name] != actual):
        raise ValueError("Changed normal input: " + name)
    hashes[name] = actual
    return actual


def view_support_matrix(points, views, policy, chunk_points=CHUNK_POINTS):
    """Same interval/voting function, bounded working arrays, one column per view."""
    points = np.asarray(points)
    if (points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all()
            or type(chunk_points) is not int or chunk_points < 1 or policy != POLICY
            or not views or len({v["view_id"] for v in views}) != len(views)):
        raise ValueError("Finite points, distinct views, fixed policy and positive chunk required")
    matrix = np.empty((len(points), len(views)), dtype=np.uint8)
    single = dict(policy, minimum_views=1)
    for begin in range(0, len(points), chunk_points):
        end = min(begin + chunk_points, len(points))
        batch = points[begin:end]
        for column, view in enumerate(views):
            matrix[begin:end, column] = support_votes(batch, [view], single)
        if not np.array_equal(matrix[begin:end].sum(axis=1), support_votes(batch, views, policy)):
            raise ValueError("Per-view sum differs from the original joint voting function")
    return matrix


def point_id_schema(ids, frames):
    """Only decode the explicit point-snapshot frame,row,column contract."""
    ids = np.asarray(ids)
    actual = dict(dtype=ids.dtype.str, shape=list(ids.shape))
    if ids.ndim != 2 or ids.shape[1:] != (3,) or ids.dtype.kind not in "ui" or np.any(ids < 0):
        return dict(state="unknown", reason="not_nonnegative_integer_Nx3", actual=actual), None
    if (not frames or any(set(f) != {"frame_id", "image_sha256", "prediction_size_wh"} for f in frames)
            or len({f["frame_id"] for f in frames}) != len(frames)):
        return dict(state="unknown", reason="missing_or_ambiguous_frame_metadata", actual=actual), None
    for frame in frames:
        size = frame["prediction_size_wh"]
        if len(size) != 2 or any(type(n) is not int or n < 1 for n in size):
            return dict(state="unknown", reason="invalid_prediction_raster", actual=actual), None
    if len(ids):
        if np.any(ids[:, 0] >= len(frames)):
            return dict(state="unknown", reason="source_frame_out_of_range", actual=actual), None
        sizes = np.asarray([f["prediction_size_wh"] for f in frames])[ids[:, 0].astype(np.int64)]
        if np.any(ids[:, 1] >= sizes[:, 1]) or np.any(ids[:, 2] >= sizes[:, 0]):
            return dict(state="unknown", reason="source_pixel_out_of_range", actual=actual), None
    return dict(state="decoded", columns=["frame_index", "row", "column"],
                frame_index_reference="snapshot.metadata.frames order", actual=actual), ids[:, 0].astype(np.int64)


def spatial_summary(points):
    """Raw XYZ extent and exact within-set nearest neighbor, not fitted geometry."""
    values = np.asarray(points, float)
    if values.ndim != 2 or values.shape[1:] != (3,) or not np.isfinite(values).all():
        raise ValueError("Finite Nx3 geometry required")
    count = len(values)
    extent = None if not count else dict(min_xyz_m=values.min(axis=0).tolist(),
        max_xyz_m=values.max(axis=0).tolist(), span_xyz_m=np.ptp(values, axis=0).tolist())
    nearest = dict(state="unavailable", reason="fewer_than_two_points", quantiles=None)
    if count >= 2:
        distances = cKDTree(values).query(values, k=2, workers=1)[0][:, 1]
        nearest = dict(state="complete", reason=None,
            quantiles={str(q): float(v) for q, v in zip(QUANTILES, np.quantile(distances, QUANTILES))},
            zero_distance_count=int(np.count_nonzero(distances == 0.)))
    return dict(point_count=count, extent=extent, nearest_neighbor_m=nearest,
                nearest_neighbor_scope="within this exact point subset; self excluded; coincident other IDs retained")


def support_summary(points, ids, frames, matrix, expected_histogram, expected_count):
    points, ids, matrix = np.asarray(points), np.asarray(ids), np.asarray(matrix)
    if (len(points) != len(ids) or matrix.shape != (len(points), len(frames))
            or matrix.dtype != np.uint8 or np.any(matrix > 1)):
        raise ValueError("Point identity or per-view support matrix shape/value mismatch")
    votes = matrix.sum(axis=1)
    histogram = np.bincount(votes.astype(np.int64), minlength=len(frames) + 1).tolist()
    selected = votes >= POLICY["minimum_views"]
    if histogram != expected_histogram or int(selected.sum()) != expected_count:
        raise ValueError("Recomputed original support mask differs from frozen normal record")
    schema, sources = point_id_schema(ids, frames)
    patterns, counts = np.unique(matrix, axis=0, return_counts=True)
    rows = []
    if sources is not None:
        for index, frame in enumerate(frames):
            source = sources == index
            mask = source & selected
            rows.append(dict(frame_index=index, frame_id=frame["frame_id"],
                prediction_size_wh=frame["prediction_size_wh"], full_point_count=int(source.sum()),
                vote_histogram=np.bincount(votes[source].astype(np.int64), minlength=len(frames)+1).tolist(),
                supported_vote_histogram=np.bincount(votes[mask].astype(np.int64), minlength=len(frames)+1).tolist(),
                mask_geometry=spatial_summary(points[mask])))
    return dict(point_id_schema=schema, view_order=[f["frame_id"] for f in frames],
        full_point_count=len(points), supported_point_count=int(selected.sum()), vote_histogram=histogram,
        per_view_supported_count=matrix.sum(axis=0).tolist(),
        support_matrix=dict(dtype=matrix.dtype.str, shape=list(matrix.shape), order="C; source snapshot point order",
                            sha256=hashlib.sha256(matrix.tobytes()).hexdigest()),
        support_patterns=[dict(per_view_support=p.tolist(), point_count=int(n)) for p, n in zip(patterns, counts)],
        sources=rows, supported_geometry=spatial_summary(points[selected]),
        support_policy=dict(POLICY), frozen_histogram_and_count_equal=True)


def fold_summary(record):
    gates, reasons = Counter(), Counter()
    for audit in record.get("ridge_fit_audits", []):
        gates.update(audit.get("gate_counts", {}))
        reasons[audit.get("sampling_scale", {}).get("reason") or "none"] += 1
    return dict(unique_fits=record.get("unique_fits"), accepted_ridge_runs=record.get("accepted_ridge_runs"),
        training_points=record.get("local_proposal_evidence", {}).get("training_points"),
        gate_counts=dict(sorted(gates.items())), sampling_reason_counts=dict(sorted(reasons.items())),
        measured_run_count=sum(len(a.get("measured_runs", [])) for a in record.get("ridge_fit_audits", [])))


def normal_summary(record):
    keys = ("case_id", "fraction", "state", "resolution_state", "reason", "full_input_point_count",
            "supported_base_point_count", "occupied_voxels", "training_points", "validation_points",
            "base_vote_histogram", "raw_row_states", "section_evidence")
    return dict(**{key: record.get(key) for key in keys}, segment_count=len(record["segments"]),
        folds=[dict(partition=0, **fold_summary(record))] +
              [dict(partition=part, **fold_summary(fold)) for part, fold in zip(
                  record.get("fold_consensus", {}).get("fixed_additional_partitions", []),
                  record.get("fold_consensus", {}).get("fold_results", []))])


def validated_inputs():
    """No evaluation/post calls: verify complete normal inventories and all bytes."""
    before, config = fixture.checked()
    normal, graphs = fixture.verified_normal(config)
    hashes = dict(before["hashes"])
    for path in (fixture.RUN / "pre.json", fixture.RUN / "inference.json",
                 fixture.previous.RUN / "pre.json", fixture.previous.RUN / "inference.json"):
        bind(path, hashes)
    for row in normal["rows"]:
        bind(fixture.RUN / row["path"], hashes, row["sha256"])
        bind(fixture.previous.RUN / row["previous_path"], hashes, row["previous_sha256"])
    point_before, manifest, _ = fixture.parent.checked()
    point_normal = fixture.read(fixture.parent.RUN / "inference.json")
    if (point_normal["run_id"] != fixture.parent.RUN_ID or point_normal["state"] != "complete"
            or point_normal["gt_read"] or point_normal["pre_sha256"] != file_hash(fixture.parent.RUN / "pre.json")
            or [r["case_id"] for r in point_normal["rows"]] != config["case_ids"]
            or [r["case_id"] for r in manifest["cases"]] != config["case_ids"]):
        raise ValueError("Incomplete or detached original point-patch normal inventory")
    for inventory in (point_before["hashes"], point_normal["outputs"]):
        for name, sha in inventory.items():
            bind(ROOT / name, hashes, sha)
    for path in (fixture.parent.RUN / "pre.json", fixture.parent.RUN / "inference.json",
                 fixture.parent.INPUTS / "manifest.json", Path(__file__), ROOT / "tests/test_g1_base_support.py"):
        bind(path, hashes)
    base_rows = [graph for graph, _ in graphs if graph["variant"] == "base"]
    if [(r["case_id"], r["fraction"]) for r in base_rows] != [
            (cid, scale) for cid in config["case_ids"] for scale in config["voxel_camera_span_fractions"]]:
        raise ValueError("Exact six base normal rows required")
    return hashes, manifest, point_normal, base_rows


def exclusive_publish(path, payload):
    encoded = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)


def run():
    if PUBLIC.exists():
        raise FileExistsError("Preserve previous base-support diagnostic: " + str(PUBLIC))
    started = time.perf_counter()
    hashes, manifest, point_normal, normals = validated_inputs()
    verification_seconds = time.perf_counter() - started
    point_rows = {row["case_id"]: row for row in point_normal["rows"]}
    cases = []
    for case in manifest["cases"]:
        tick = time.perf_counter()
        cid, folder = case["case_id"], fixture.parent.RUN / case["case_id"]
        source_path, cameras_path = ROOT / case["parent_record"], folder / "rods.json"
        bind(source_path, hashes, case["parent_sha256"])
        bind(cameras_path, hashes, point_normal["outputs"][cameras_path.relative_to(ROOT).as_posix()])
        raw, rods = fixture.read(source_path), fixture.read(cameras_path)
        views = support_views(raw["frames"], rods["cameras"])
        snapshot, arrays = load_snapshot(folder / "bundle/base")
        frames, meta, original = snapshot["metadata"]["frames"], snapshot["metadata"], point_rows[cid]
        if (snapshot["content_id"] != original["snapshot_id"] or len(arrays["points"]) != original["full_point_count"]
                or meta["length_unit"] != "meter" or meta["source_prediction_sha256"] != original["prediction_sha256"]
                or meta["world_frame_id"] != "fixture-conditioned-prediction:" + original["prediction_sha256"]
                or [f["frame_id"] for f in frames] != [v["view_id"] for v in views]
                or [f["image_sha256"] for f in frames] != [f["rgb_sha256"] for f in case["frames"]]):
            raise ValueError("Snapshot source frame, image or normal identity differs")
        matrix = view_support_matrix(arrays["points"], views, POLICY)
        expected = [r for r in normals if r["case_id"] == cid]
        if (len(expected) != 2 or expected[0]["base_vote_histogram"] != expected[1]["base_vote_histogram"]
                or expected[0]["supported_base_point_count"] != expected[1]["supported_base_point_count"]
                or any(r["full_input_point_count"] != len(arrays["points"]) for r in expected)):
            raise ValueError("Original paired-scale base support differs")
        summary = support_summary(arrays["points"], arrays["point_ids"], frames, matrix,
            expected[0]["base_vote_histogram"], expected[0]["supported_base_point_count"])
        summary.update(case_id=cid, snapshot_id=snapshot["content_id"],
            raw_row_states=[dict(view_id=v["view_id"], **v["row_states"]) for v in views],
            elapsed_seconds=time.perf_counter()-tick)
        cases.append(summary)
        print("G1_BASE_SUPPORT", cid, summary["full_point_count"], summary["supported_point_count"], flush=True)
        del matrix, arrays
    tick = time.perf_counter()
    verify_receipts(hashes)
    final_verification_seconds = time.perf_counter() - tick
    report = dict(schema_version="g1-base-support-diagnostic-v1", created_at_utc=datetime.now(timezone.utc).isoformat(),
        state="complete", gt_read=False, normal_run_id=fixture.RUN_ID, point_run_id=fixture.parent.RUN_ID,
        source_sha256=file_hash(__file__), input_sha256=dict(sorted(hashes.items())),
        normal_rows=[normal_summary(row) for row in normals], cases=cases,
        runtime=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                     platform=platform.platform(), executable=sys.executable),
        cost=dict(elapsed_seconds_before_publication=time.perf_counter()-started,
                  initial_verification_seconds=verification_seconds, final_verification_seconds=final_verification_seconds,
                  chunk_points=CHUNK_POINTS, peak_memory_measured=False,
                  memory_policy="One full base and Nx5 uint8 support matrix at a time; bounded projection batches; exact KDTree only on supported subsets"),
        method_execution=dict(new_readout=False, new_depth=False, new_camera=False, policy_changed=False),
        conclusion=dict(state="unresolved", physical_base_contains_rod="not_established_by_this_diagnostic",
            reader_missed_real_rod="not_established_by_this_diagnostic",
            evidence="Frozen normals found no supported finite straight model after the existing RGB mask; this report adds source counts and raw spatial spacing only",
            unresolved_causes=["source depth geometry", "estimated camera/projection error", "RGB support-mask selection", "reader model or finite-support rejection"],
            scope="Development normal-data diagnosis; neither truth qualification, base-absence proof, foreground identity nor G1 pass"))
    exclusive_publish(PUBLIC, report)
    print("G1_BASE_SUPPORT_REPORT", PUBLIC, len(hashes), "verified receipts", flush=True)


if __name__ == "__main__":
    from environment_paths import require_project_environment

    require_project_environment(ROOT)
    sys.addaudithook(block_truth)
    run()
