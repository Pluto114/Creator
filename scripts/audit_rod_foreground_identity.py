"""Post-evaluation audit: annotation visibility and retained-candidate failure causes.

GT is used here only after frozen inference. These labels never repair clicks or
choose predictions; an apparent RGB band can span empty background between rods.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from run_rod_candidate_ablation import checked_artifact
from run_rod_foreground_identity import checked_run, profile_anchors
from run_rod_identity_blender import digest, locations, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def touched_pixels(center, uncertainty):
    low = int(np.ceil(center - uncertainty - .5 + 1e-9))
    high = int(np.floor(center + uncertainty + .5 - 1e-9))
    return list(range(low, high + 1)) if high >= low else [int(np.floor(center + .5))]


def audit(config_path, summary_path):
    config, summary = read_json(config_path), read_json(summary_path)
    run, _, prepared, method, _, annotations = checked_run(config["run_id"])
    if digest(run / "inference.json") != summary["inference_sha256"] or digest(config_path) != summary["protocol_sha256"]:
        raise ValueError("Audit refers to different frozen inference")
    _, _, truth, _ = locations(prepared["parent_run_id"])
    targets = {c["case_id"]: read_json(truth / c["path"]) for c in read_json(truth / "manifest.json")["cases"]}
    queries = {q["query_id"]: q for q in config["evaluation_queries"]}
    audits = []
    for query in annotations["queries"]:
        target = targets[query["case_id"]]
        selector = queries[query["query_id"]]["target_selector"]
        expected = (1 if target["target"]["present"] else None) if selector == "declared" else (None if selector == "empty" else selector)
        frames = {f["view_id"]: f for f in target["frames"]}
        for profile in method["profiles"]:
            checks = []
            for a in profile_anchors(query, profile):
                artifact = frames[a["view_id"]]["arrays"]["surface_id"]
                path = truth / artifact["path"]
                if digest(path) != artifact["sha256"]:
                    raise ValueError("Truth surface raster changed")
                surface = np.load(path, allow_pickle=False)
                xs, ys = [touched_pixels(c, u) for c, u in zip(a["xy"], a["uncertainty_xy_px"])]
                values = surface[np.ix_(ys, xs)].reshape(-1)
                checks.append(dict(view_id=a["view_id"], xy=a["xy"], touched_pixel_count=len(values), surface_id_counts={str(k): v for k, v in Counter(map(int, values)).items()},
                                   all_pixel_centers_on_requested_object=expected is not None and bool(np.all(values == expected))))
            audits.append(dict(query_id=query["query_id"], profile=profile["name"], expected_surface_id=expected, anchors=checks))
    lookup = {(a["query_id"], a["profile"]): a for a in audits}
    errors = [{**r, "annotation_visibility": lookup[r["query_id"], r["profile"]]} for r in summary["rows"] if r["classification"] in ("wrong_line_accept", "false_accept_empty")]
    failures = []
    for entry in read_json(run / "inference.json")["records"]:
        record = checked_artifact(run, entry)
        for variant in record["rows"]:
            if variant["profile"] != "two_clicks" or variant["identity"]["state"] == "accepted":
                continue
            result = variant["identity"]
            proposals = result["proposal_audit"]
            reasons = Counter()
            for p in proposals:
                for a in p.get("anchors", []):
                    if a["state"] != "supported":
                        reasons[a.get("reason", "anchor_" + a["state"])] += 1
                    for row in a.get("rows", []):
                        if row.get("reason"):
                            reasons[row["reason"]] += 1
            failures.append({**{k: record[k] for k in ("case_id", "cap", "method", "search_offset_px")},
                "query_id": variant["query_id"], "state": result["state"], "reason": result["reason"],
                "proposal_states": dict(Counter(p["state"] for p in proposals)), "anchor_failure_counts": dict(reasons)})
    return dict(run_id=config["run_id"], summary_sha256=digest(summary_path), annotations=audits, false_outputs=errors, primary_profile_refusal_causes=failures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiments/results/2026-09-21-foreground-audit.json")
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    result = dict(scope="evaluation-only pixel-center visibility; not antialiased RGB purity or an inference feature", source_sha256=digest(Path(__file__)), runs=[audit(ROOT / f"configs/rod_foreground_{config}_v1.json", ROOT / f"docs/experiments/results/2026-09-21-foreground-{summary}.json") for config, summary in (("development", "development"), ("new_queries", "new-queries"))])
    write_json(args.output, result)
    print("AUDITED_FOREGROUND", [(r["run_id"], len(r["false_outputs"])) for r in result["runs"]])
