"""Bound one accepted RGB association with conservative, per-row finite evidence.

No truth, visibility masks, or target endpoints enter this adapter. A successful
result is conditional on the upstream identity decision; it is not a certified
physical axis or a formal PatchResult.
"""

from __future__ import annotations

import numpy as np

from .rod_evidence import build_rod_candidate


def _index(value, length, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer index")
    if not 0 <= value < length:
        raise ValueError(f"{name} is outside its source collection")
    return int(value)


def bound_selected_candidate(association_result, views, observations_by_view, config=None):
    """Turn only the selected image hypotheses' measured rows into finite support.

    ``views`` use the association input format plus ``size_wh``. Raw observation
    packages must be supplied in the same order. Each selected hypothesis contains
    ``row_matches`` triples: (raw row index, raw candidate index, residual).

    All unmatched rows, including flat/absent detector responses, remain unknown.
    We have no independent visibility proof here, so a flat occluder must not turn
    into a negative vote. The existing finite stage keeps fixed cameras, refits
    only the selected 2D lines, and requires four fit views for leave-one-out.
    """
    metadata = {
        "association_state": association_result.get("state"),
        "scope": "conditional_finite_rgb_diagnostic_not_physical_identity_or_formal_patch",
        "observation_policy": "selected row_matches only; all other rows unknown; no negative evidence",
        "line_policy": "refit selected supporting image lines with fixed input cameras and leave-one-out gate",
    }
    if association_result.get("state") != "accepted":
        return {
            **metadata,
            "state": "rejected",
            "segments": np.empty((0, 2, 3)),
            "shadow_segments": np.empty((0, 2, 3)),
            "rejection_reasons": ["association_not_accepted"],
            "candidate_selection": [],
        }
    if len(views) != len(observations_by_view):
        raise ValueError("Raw observations must match the view order and count")
    selected = association_result.get("selected")
    if not isinstance(selected, dict) or len(selected.get("matches", [])) != len(views):
        raise ValueError("Accepted association must contain one match record per view")
    supporting = [
        _index(index, len(views), "supporting view")
        for index in selected["supporting_views"]
    ]
    if len(set(supporting)) != len(supporting):
        raise ValueError("Supporting view indices must be unique")
    finite_views, selections = [], []
    for view_index, (view, extracted) in enumerate(zip(views, observations_by_view)):
        rows = extracted["rows"]
        row_choices = {}
        hypothesis = None
        candidate_index = None
        if view_index in supporting:
            candidate_index = _index(
                selected["matches"][view_index]["candidate_index"],
                len(view["candidates"]), "selected image candidate",
            )
            hypothesis = view["candidates"][candidate_index]
            for match in hypothesis["row_matches"]:
                if len(match) != 3:
                    raise ValueError("Expected row_matches triples")
                row_index = _index(match[0], len(rows), "matched row")
                raw_candidate_index = _index(
                    match[1], len(rows[row_index]["candidates"]), "matched raw candidate"
                )
                if row_index in row_choices:
                    raise ValueError("A selected hypothesis may match each raw row only once")
                row_choices[row_index] = raw_candidate_index
        observations = []
        for row_index, row in enumerate(rows):
            item = {
                "xy": [row["guide_x"], row["y"]],
                "width_px": 0.0,
                "state": "unknown",
            }
            if row_index in row_choices:
                # 不重新挑一个更顺眼的像素：对应层选了哪一对边，就用那一对。
                candidate = row["candidates"][row_choices[row_index]]
                item.update(
                    xy=[candidate["center_x"], row["y"]],
                    width_px=candidate["width"],
                    state="accepted",
                )
            observations.append(item)
        finite_views.append({
            "view_id": view["view_id"],
            "K_index": view["K_index"],
            "world_to_camera_cv": view["world_to_camera_cv"],
            "size_wh": view["size_wh"],
            "image_line": hypothesis["line"] if hypothesis is not None else None,
            "observations": observations,
        })
        selections.append({
            "view_id": view["view_id"],
            "candidate_index": candidate_index,
            "row_matches": [[index, choice] for index, choice in sorted(row_choices.items())],
        })
    # 不见了的地方就先空着。把一串 min/max 连起来很容易，证明中间有杆很难。
    result = build_rod_candidate(
        finite_views, config, geometric_gate=True, negative_veto=False
    )
    result.update(metadata, candidate_selection=selections)
    return result
