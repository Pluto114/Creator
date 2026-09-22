"""Versioned point-snapshot and curve-patch pilot, independent of Blender and ML.

This is the small materialized-point protocol, not the full depth-backed
BaseSnapshot/PatchResult wire contract reserved elsewhere. Nothing edits the base.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import stat
from pathlib import Path

import numpy as np

VERSION = "0.1.0"
KINDS = {"point_snapshot": {"points", "point_ids"}, "curve_patch": {"suppressed_ids", "segments"}}


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_hash(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Expected a SHA256 content identity")


def ordinary_path(path):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        if item.exists() or item.is_symlink():
            info = item.lstat()
            if item.is_symlink() or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
                raise ValueError("Bundle paths cannot cross symlinks or reparse points")
    return path


def points_array(value, shape_tail):
    a = np.asarray(value, dtype="<f8")
    if a.ndim != len(shape_tail) + 1 or a.shape[1:] != shape_tail or not np.isfinite(a).all():
        raise ValueError("Finite geometry with the declared shape required")
    return np.ascontiguousarray(a)


def ids_array(value):
    a = np.asarray(value)
    if a.ndim != 2 or a.shape[1] != 3 or a.dtype.kind not in "ui" or np.any(a < 0) or np.any(a > np.iinfo(np.uint32).max):
        raise ValueError("Point IDs must be nonnegative integer [N,3] arrays")
    a = np.ascontiguousarray(a, dtype="<u4")
    if len(a):
        order = np.lexsort(a.T[::-1])
        if not np.array_equal(order, np.arange(len(a))) or np.any(np.all(a[1:] == a[:-1], axis=1)):
            raise ValueError("Point IDs must be sorted and unique; preview indices are not IDs")
    return a


def validate_frames(frames):
    if not frames or len({f["frame_id"] for f in frames}) != len(frames):
        raise ValueError("Unique nonempty frame order required")
    for f in frames:
        if set(f) != {"frame_id", "image_sha256", "prediction_size_wh"} or not isinstance(f["frame_id"], str) or not f["frame_id"]:
            raise ValueError("Invalid frame declaration")
        require_hash(f["image_sha256"])
        if len(f["prediction_size_wh"]) != 2 or any(type(n) is not int or not 0 < n <= np.iinfo(np.uint32).max for n in f["prediction_size_wh"]):
            raise ValueError("Positive prediction raster size required")


def _publish(destination, kind, metadata, arrays):
    destination = ordinary_path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    refs = {}
    for name, array in arrays.items():
        path = destination / (name + ".npy")
        with path.open("xb") as stream:
            np.save(stream, array, allow_pickle=False)
        refs[name] = {"file": path.name, "sha256": file_hash(path), "dtype": array.dtype.str, "shape": list(array.shape)}
    manifest = {"schema_version": VERSION, "kind": kind, "metadata": copy.deepcopy(metadata), "arrays": refs}
    manifest["content_id"] = content_hash(manifest)
    # 先占用新目录，最后发布收据。中途失败的半包留下排查，读取器绝不当成功。
    pending = destination / "manifest.pending.json"
    with pending.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
    pending.rename(destination / "manifest.json")
    return manifest["content_id"]


def load_bundle(directory, expected_kind):
    directory = ordinary_path(directory)
    path = ordinary_path(directory / "manifest.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if set(manifest) != {"schema_version", "kind", "metadata", "arrays", "content_id"} or manifest["schema_version"] != VERSION or manifest["kind"] != expected_kind:
        raise ValueError("Unsupported point/patch bundle contract")
    require_hash(manifest["content_id"])
    body = {k: v for k, v in manifest.items() if k != "content_id"}
    if content_hash(body) != manifest["content_id"] or set(manifest["arrays"]) != KINDS[expected_kind]:
        raise ValueError("Bundle identity or artifact inventory changed")
    arrays = {}
    for name, ref in manifest["arrays"].items():
        if set(ref) != {"file", "sha256", "dtype", "shape"} or ref["file"] != name + ".npy":
            raise ValueError("Artifact references must be canonical local filenames")
        require_hash(ref["sha256"])
        path = ordinary_path(directory / ref["file"])
        if file_hash(path) != ref["sha256"]:
            raise ValueError("Artifact bytes changed")
        a = np.load(path, allow_pickle=False)
        if a.dtype.str != ref["dtype"] or list(a.shape) != ref["shape"]:
            raise ValueError("Artifact shape or dtype changed")
        arrays[name] = a
    return manifest, arrays


def validate_snapshot(m, a):
    meta = m["metadata"]
    if set(meta) != {"frames", "world_frame_id", "length_unit", "source_prediction_sha256", "point_policy"}:
        raise ValueError("Snapshot metadata fields changed")
    validate_frames(meta["frames"])
    require_hash(meta["source_prediction_sha256"])
    if not isinstance(meta["world_frame_id"], str) or not meta["world_frame_id"] or meta["length_unit"] not in ("meter", "reconstruction_unit"):
        raise ValueError("Known coordinate identity and unit required")
    if not isinstance(meta["point_policy"], str) or not meta["point_policy"]:
        raise ValueError("Materialization policy required")
    points, ids = points_array(a["points"], (3,)), ids_array(a["point_ids"])
    if a["points"].dtype.str != "<f8" or a["point_ids"].dtype.str != "<u4" or len(points) != len(ids):
        raise ValueError("Unexpected point payload")
    if len(ids):
        if int(ids[:, 0].max()) >= len(meta["frames"]):
            raise ValueError("Unknown source frame index")
        sizes = np.asarray([f["prediction_size_wh"] for f in meta["frames"]], dtype=np.int64)[ids[:, 0]]
        if np.any(ids[:, 1] >= sizes[:, 1]) or np.any(ids[:, 2] >= sizes[:, 0]):
            raise ValueError("Point ID is outside its source raster")
    return m, {"points": points, "point_ids": ids}


def load_snapshot(directory):
    m, a = load_bundle(directory, "point_snapshot")
    return validate_snapshot(m, a)


def write_snapshot(destination, points, point_ids, *, frames, world_frame_id, length_unit, source_prediction_sha256, point_policy):
    validate_frames(frames)
    require_hash(source_prediction_sha256)
    a = {"points": points_array(points, (3,)), "point_ids": ids_array(point_ids)}
    if len(a["points"]) != len(a["point_ids"]):
        raise ValueError("Every base point needs its actual frame/row/column identity")
    metadata = dict(frames=frames, world_frame_id=world_frame_id, length_unit=length_unit, source_prediction_sha256=source_prediction_sha256, point_policy=point_policy)
    validate_snapshot({"metadata": metadata}, a)
    return _publish(destination, "point_snapshot", metadata, a)


def _suppression_indices(base_ids, suppressed):
    # 查的是(frame,row,col)，不是当前显示顺序。不存在的点不能糊弄成“已经删掉”。
    dtype = np.dtype([("frame", "<u4"), ("row", "<u4"), ("col", "<u4")])
    source, requested = base_ids.view(dtype).reshape(-1), suppressed.view(dtype).reshape(-1)
    indices = np.searchsorted(source, requested)
    if np.any(indices >= len(source)) or not np.array_equal(source[indices], requested):
        raise ValueError("Suppression references a point absent from the bound snapshot")
    return indices


def validate_patch(snapshot, base_arrays, manifest, arrays):
    meta = manifest["metadata"]
    fields = {"base_snapshot_id", "world_frame_id", "length_unit", "selection_sha256", "method", "outcome", "line_ids", "evidence", "unresolved"}
    if set(meta) != fields:
        raise ValueError("Patch metadata fields changed")
    if meta["base_snapshot_id"] != snapshot["content_id"] or any(meta[k] != snapshot["metadata"][k] for k in ("world_frame_id", "length_unit")):
        raise ValueError("Stale patch or coordinate-frame mismatch")
    require_hash(meta["selection_sha256"])
    if set(meta["method"]) != {"id", "version", "config_sha256", "seed"} or not meta["method"]["id"] or not meta["method"]["version"] or type(meta["method"]["seed"]) is not int:
        raise ValueError("Named method and exact config/seed required")
    require_hash(meta["method"]["config_sha256"])
    segments, suppressed = points_array(arrays["segments"], (2, 3)), ids_array(arrays["suppressed_ids"])
    if arrays["segments"].dtype.str != "<f8" or arrays["suppressed_ids"].dtype.str != "<u4":
        raise ValueError("Unexpected patch dtype")
    if np.any(np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1) <= 1e-10):
        raise ValueError("Zero-length added segment")
    line_ids = meta["line_ids"]
    if len(line_ids) != len(segments) or len(set(line_ids)) != len(line_ids) or any(not isinstance(v, str) or not v for v in line_ids):
        raise ValueError("Every finite segment needs a unique line ID")
    changed = bool(len(segments) or len(suppressed))
    if meta["outcome"] != ("accepted_change" if changed else "no_supported_change"):
        raise ValueError("Patch outcome contradicts its actual geometry")
    known_views = {f["frame_id"] for f in snapshot["metadata"]["frames"]}
    covered_lines, covered_suppression = [], []
    for evidence in meta["evidence"]:
        if set(evidence) != {"decision", "line_ids", "suppression_range", "view_ids", "source_sha256", "note"}:
            raise ValueError("Evidence contract mismatch")
        require_hash(evidence["source_sha256"])
        if evidence["decision"] not in ("accept", "reject", "unresolved") or not isinstance(evidence["note"], str) or not evidence["note"]:
            raise ValueError("Explicit evidence decision and note required")
        if len(set(evidence["view_ids"])) != len(evidence["view_ids"]) or not set(evidence["view_ids"]) <= known_views:
            raise ValueError("Evidence references duplicate/unknown views")
        ids, interval = evidence["line_ids"], evidence["suppression_range"]
        if evidence["decision"] != "accept" and (ids or interval is not None):
            raise ValueError("Rejected/unknown evidence cannot authorize geometry")
        if ids or interval is not None:
            if not evidence["view_ids"]:
                raise ValueError("Changes require source observations")
        if not set(ids) <= set(line_ids):
            raise ValueError("Evidence refers to absent added geometry")
        covered_lines.extend(ids)
        if interval is not None:
            if len(interval) != 2 or any(type(i) is not int for i in interval) or not 0 <= interval[0] < interval[1] <= len(suppressed):
                raise ValueError("Evidence suppression range is invalid")
            covered_suppression.extend(range(*interval))
    if sorted(covered_lines) != sorted(line_ids) or sorted(covered_suppression) != list(range(len(suppressed))):
        raise ValueError("Each change needs exactly one explicit accepting evidence record")
    if not isinstance(meta["unresolved"], list) or any(not isinstance(r, str) or not r for r in meta["unresolved"]):
        raise ValueError("Unresolved reasons must be explicit strings")
    return _suppression_indices(base_arrays["point_ids"], suppressed)


def write_patch(destination, snapshot_dir, segments, suppressed_ids, *, selection_sha256, method, evidence, unresolved):
    snapshot, base = load_snapshot(snapshot_dir)
    segments, suppressed_ids = points_array(segments, (2, 3)), ids_array(suppressed_ids)
    meta = dict(base_snapshot_id=snapshot["content_id"], world_frame_id=snapshot["metadata"]["world_frame_id"], length_unit=snapshot["metadata"]["length_unit"],
        selection_sha256=selection_sha256, method=method, outcome="accepted_change" if len(segments) or len(suppressed_ids) else "no_supported_change",
        line_ids=[f"line-{i:04d}" for i in range(len(segments))], evidence=evidence, unresolved=unresolved)
    arrays = dict(segments=segments, suppressed_ids=suppressed_ids)
    validate_patch(snapshot, base, {"metadata": meta}, arrays)
    return _publish(destination, "curve_patch", meta, arrays)


def compose(snapshot_dir, patch_dir=None, *, enabled=True):
    if type(enabled) is not bool:
        raise ValueError("Composition enabled state must be boolean")
    snapshot, base = load_snapshot(snapshot_dir)
    patch_id, segments = None, np.empty((0, 2, 3), dtype="<f8")
    keep = np.ones(len(base["points"]), dtype=bool)
    if patch_dir is not None:
        patch, arrays = load_bundle(patch_dir, "curve_patch")
        indices = validate_patch(snapshot, base, patch, arrays)
        patch_id = patch["content_id"]
        if enabled:
            keep[indices] = False
            segments = arrays["segments"].copy()
    # 撤回只是不应用补丁，原点的位置和身份必须逐位回来；不靠再拟合“差不多恢复”。
    return dict(base_snapshot_id=snapshot["content_id"], patch_id=patch_id, enabled=bool(enabled),
                points=base["points"][keep], point_ids=base["point_ids"][keep], segments=segments,
                world_frame_id=snapshot["metadata"]["world_frame_id"], length_unit=snapshot["metadata"]["length_unit"])


def write_candidate_view(path, snapshot_dir, patch_dir=None, *, enabled=True):
    """Save an immutable view choice; enabling/disabling never rewrites either bundle."""
    path = ordinary_path(path)
    if type(enabled) is not bool:
        raise ValueError("View enabled state must be boolean")
    references = {}
    for key, directory in (("snapshot", snapshot_dir), ("patch", patch_dir)):
        if directory is None:
            references[key] = None
            continue
        relative = ordinary_path(directory).relative_to(path.parent)
        if len(relative.parts) != 1 or re.fullmatch(r"[A-Za-z0-9_-]+", relative.as_posix()) is None:
            raise ValueError("Candidate view references canonical sibling bundle directories only")
        references[key] = relative.as_posix()
    candidate = compose(snapshot_dir, patch_dir, enabled=enabled)
    body = dict(schema_version=VERSION, kind="point_candidate_view", references=references,
                base_snapshot_id=candidate["base_snapshot_id"], patch_id=candidate["patch_id"], enabled=enabled)
    body["content_id"] = content_hash(body)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(body, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return body["content_id"]


def open_candidate_view(path):
    path = ordinary_path(path)
    body = json.loads(path.read_text(encoding="utf-8"))
    if set(body) != {"schema_version", "kind", "references", "base_snapshot_id", "patch_id", "enabled", "content_id"} or body["schema_version"] != VERSION or body["kind"] != "point_candidate_view" or type(body["enabled"]) is not bool:
        raise ValueError("Candidate view contract mismatch")
    if content_hash({k: v for k, v in body.items() if k != "content_id"}) != body["content_id"]:
        raise ValueError("Candidate view identity changed")
    if set(body["references"]) != {"snapshot", "patch"} or body["references"]["snapshot"] is None:
        raise ValueError("Candidate view needs an exact base reference")
    directories = {}
    for key, relative in body["references"].items():
        if relative is None:
            directories[key] = None
        elif not isinstance(relative, str) or re.fullmatch(r"[A-Za-z0-9_-]+", relative) is None:
            raise ValueError("Candidate view cannot leave its bundle directory")
        else:
            directories[key] = path.parent / relative
    candidate = compose(directories["snapshot"], directories["patch"], enabled=body["enabled"])
    if any(candidate[k] != body[k] for k in ("base_snapshot_id", "patch_id")):
        raise ValueError("Candidate view references replaced bundle content")
    return candidate
