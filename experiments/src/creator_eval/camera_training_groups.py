"""Fixed RGB-training cell leave-group plans, independent of cameras and GT."""
from __future__ import annotations

import copy
import hashlib
import json
import math

CELL_SIZE_PX = 64
GROUP_COUNT = 4
DEFAULT_SEED = 24092026


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _normalize_training(training, view_count):
    _integer(view_count, "view_count", 3)
    normalized, seen = [], set()
    for track in training:
        tid = _integer(track["track_id"], "track_id")
        if tid in seen:
            raise ValueError("Training track ids must be unique")
        seen.add(tid)
        observations, views = [], set()
        for observation in track["observations"]:
            view = _integer(observation["view"], "view")
            if view >= view_count or view in views:
                raise ValueError("Distinct valid observation views required")
            views.add(view)
            xy = list(observation["xy"])
            if len(xy) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in xy):
                raise ValueError("Finite two-dimensional RGB coordinates required")
            observations.append(dict(view=view, xy=[float(v) for v in xy]))
        if len(observations) < 2:
            raise ValueError("A training track needs at least two observed views")
        normalized.append(dict(track_id=tid, observations=sorted(observations, key=lambda o: o["view"])))
    return sorted(normalized, key=lambda t: t["track_id"])


def build_training_group_plan(training, *, validation_track_ids, view_count,
                              seed=DEFAULT_SEED, minimum_tracks=24, minimum_tracks_per_view=16):
    """Assign complete tracks using their earliest observed view's 64px cell.

    Only training coordinates are accepted. Validation ids are used solely to
    reject overlap; neither validation pixels nor any camera matrix is an input.
    The hash rule is fixed, not balanced after seeing group counts or results.
    """
    seed = _integer(seed, "seed")
    _integer(minimum_tracks, "minimum_tracks", 1)
    _integer(minimum_tracks_per_view, "minimum_tracks_per_view", 1)
    tracks = _normalize_training(training, view_count)
    train_ids = [t["track_id"] for t in tracks]
    validation_ids = [_integer(tid, "validation_track_id") for tid in validation_track_ids]
    if len(set(validation_ids)) != len(validation_ids):
        raise ValueError("Validation track ids must be unique")
    if set(train_ids) & set(validation_ids):
        raise ValueError("Training and validation tracks must be disjoint")
    assignments, cells = [], {}
    for track in tracks:
        first = track["observations"][0]
        cx, cy = [math.floor(value / CELL_SIZE_PX) for value in first["xy"]]
        cell = (first["view"], cx, cy)
        # 同一个小格子里的轨迹一起撤掉。别看到哪组不好看再偷偷换seed。
        key = f"{seed}/{cell[0]}/{cell[1]}/{cell[2]}"
        group = int.from_bytes(hashlib.sha256(key.encode("ascii")).digest()[:8], "big") % GROUP_COUNT
        assignments.append(dict(track_id=track["track_id"], cell=list(cell), group=group))
        cells.setdefault(cell, dict(cell=list(cell), group=group, track_ids=[]))["track_ids"].append(track["track_id"])
    conditions = []
    for group in (None, *range(GROUP_COUNT)):
        removed = [a["track_id"] for a in assignments if a["group"] == group] if group is not None else []
        removed_set = set(removed)
        kept = [t for t in tracks if t["track_id"] not in removed_set]
        coverage = [sum(any(o["view"] == view for o in t["observations"]) for t in kept) for view in range(view_count)]
        reasons = []
        if group is not None and not removed:
            reasons.append("empty_removed_group")
        if len(kept) < minimum_tracks:
            reasons.append("insufficient_remaining_tracks")
        if min(coverage) < minimum_tracks_per_view:
            reasons.append("insufficient_remaining_per_view_coverage")
        conditions.append(dict(condition_id="control" if group is None else f"leave_group_{group}",
                               removed_group=group, state="skipped" if reasons else "eligible", skip_reasons=reasons,
                               kept_track_ids=[t["track_id"] for t in kept], removed_track_ids=removed,
                               kept_track_count=len(kept), removed_track_count=len(removed), coverage=coverage))
    return dict(version="training-cell-leave-group-v1", training_sha256=_digest(tracks),
                training_track_ids=train_ids, validation_track_ids=sorted(validation_ids),
                policy=dict(cell_size_px=CELL_SIZE_PX, group_count=GROUP_COUNT, seed=seed, view_count=view_count,
                            minimum_tracks=minimum_tracks, minimum_tracks_per_view=minimum_tracks_per_view,
                            cell_rule="lowest observed view index, then floor(x/64), floor(y/64)",
                            group_rule="ASCII seed/view/cell_x/cell_y -> SHA256 first 8 bytes big-endian modulo 4",
                            empty_or_insufficient_group="skip condition; no redistribution or replacement tracks"),
                assignments=assignments, cells=[cells[c] for c in sorted(cells)], conditions=conditions,
                scope="Training-only spatial block sensitivity diagnostic; control plus four leave-groups; no independent validation or GT used for grouping")


def select_training_tracks(training, plan, condition_id):
    """Return complete frozen tracks in canonical order, reusable for any camera."""
    tracks = _normalize_training(training, plan["policy"]["view_count"])
    if _digest(tracks) != plan["training_sha256"]:
        raise ValueError("Training observations changed after the plan was frozen")
    matches = [c for c in plan["conditions"] if c["condition_id"] == condition_id]
    if len(matches) != 1:
        raise ValueError("One declared condition required")
    condition = matches[0]
    if condition["state"] != "eligible":
        raise ValueError("Cannot fit a skipped training condition")
    kept, removed = set(condition["kept_track_ids"]), set(condition["removed_track_ids"])
    if kept & removed or kept | removed != set(plan["training_track_ids"]):
        raise ValueError("Condition must partition complete training tracks")
    return copy.deepcopy([track for track in tracks if track["track_id"] in kept])
