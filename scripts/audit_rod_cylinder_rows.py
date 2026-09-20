"""Post-hoc pixel-ID audit of width-screen residuals; never used by inference."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.rod_cylinder_gate import (  # noqa: E402
    refit_hypothesis,
    screen_cylinder,
    silhouette_x,
)
from run_rod_candidate_ablation import checked_artifact, inference_view  # noqa: E402
from run_rod_identity_blender import digest, locations, read_json, write_json  # noqa: E402
from run_rod_support_recheck import checked_run  # noqa: E402


def audit(output):
    run, _, prepared, method, _, _ = checked_run("rod-refit-support-v1-20260920")
    inference = read_json(run / "inference.json")
    _, _, truth, _ = locations(prepared["parent_run_id"])
    truth_hashes = read_json(truth / "artifact_hashes.json")
    targets = {c["case_id"]: read_json(truth / c["path"]) for c in read_json(truth / "manifest.json")["cases"]}
    pools = {}
    for entry in inference["pools"]:
        pool = checked_artifact(run, entry)
        pools[(pool["case_id"], pool["search_offset_px"])] = pool
    rows = []
    for entry in inference["associations"]:
        record = checked_artifact(run, entry)
        if record["cap"] != 8 or record["search_offset_px"] != 0:
            continue
        frames = pools[(record["case_id"], 0)]["frames"]
        views = [inference_view(f, f["pool"]["candidates"][:8], 0) for f in frames]
        raw = [f["observations"] for f in frames]
        hypothesis = refit_hypothesis(record["association"]["selected"], views, method["extent"])
        screen = screen_cylinder(hypothesis, views, raw)
        truth_frames = {f["view_id"]: f for f in targets[record["case_id"]]["frames"]}
        for vi in hypothesis["supporting_views"]:
            view, observed = views[vi], raw[vi]
            selected = view["candidates"][hypothesis["matches"][vi]["candidate_index"]]
            pairs = [(observed["rows"][ri], observed["rows"][ri]["candidates"][ci]) for ri, ci, _ in selected["row_matches"]]
            ys = np.array([r["y"] for r, _ in pairs])
            xy = np.array([[p["left_edge"]["x"], p["right_edge"]["x"]] for _, p in pairs])
            predicted = silhouette_x(hypothesis["model"], view, screen["radius"], ys).T
            bad = np.any(np.abs(predicted - xy) > 1.0 + 1e-9, axis=1)
            relative = truth_frames[view["view_id"]]["arrays"]["surface_id"]["path"]
            if digest(truth / relative) != truth_hashes[relative]:
                raise ValueError("Frozen ID map changed")
            surface = np.load(truth / relative, allow_pickle=False)
            centers = np.rint([[p["center_x"], r["y"]] for r, p in pairs]).astype(int)
            if np.any(centers < 0) or np.any(centers[:, 0] >= surface.shape[1]) or np.any(centers[:, 1] >= surface.shape[0]):
                raise ValueError("Measured row center outside image")
            labels = surface[centers[:, 1], centers[:, 0]]
            bad_ys = ys[bad].astype(int)
            intervals = np.split(bad_ys, np.where(np.diff(bad_ys) > 1)[0] + 1)
            rows.append({"case_id": record["case_id"], "view_id": view["view_id"], "matched_rows": len(pairs),
                         "bad_rows": int(bad.sum()), "bad_row_intervals": [[int(a[0]), int(a[-1])] for a in intervals if len(a)],
                         "bad_center_ids": {str(k): v for k, v in Counter(labels[bad].tolist()).items()},
                         "good_center_ids": {str(k): v for k, v in Counter(labels[~bad].tolist()).items()}})
    write_json(output, {"state": "complete", "truth_read": True, "predictions_changed": False,
                        "scope": "post-hoc explanation only; all five default-window cap8 best assignments, never a selection feature",
                        "source_sha256": digest(Path(__file__)), "inference_sha256": digest(run / "inference.json"),
                        "truth_artifacts_sha256": digest(truth / "artifact_hashes.json"), "rows": rows,
                        "limitations": "Rounded center ID is not proof that paired edges are silhouettes; antialias sampling differs."})
    print("AUDITED_CYLINDER_ROWS", len(rows), "view assignments")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiments/results/2026-09-20-cylinder-row-audit.json")
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    audit(args.output)
