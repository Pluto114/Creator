"""Audit final candidate support on frozen observation pools, without ground truth."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "configs").is_dir() and (parent / "experiments/src/creator_eval").is_dir())


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def no_truth_reads(event, arguments):
    if event == "open" and isinstance(arguments[0], (str, bytes)):
        value = arguments[0].decode() if isinstance(arguments[0], bytes) else arguments[0]
        path = Path(value).resolve()
        if any(path.is_relative_to(ROOT / relative) for relative in ("data/eval_gt", "data/evaluation")):
            raise PermissionError("This support audit must not access ground truth or evaluation files")


def counts(candidates):
    return {
        "candidate_count": len(candidates),
        "rows_below_minimum": sum(row["rows_below_minimum"] for row in candidates),
        "span_below_minimum": sum(row["span_below_minimum"] for row in candidates),
        "either_below_minimum": sum(row["rows_below_minimum"] or row["span_below_minimum"] for row in candidates),
    }


def audit(run_id, output):
    if not run_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Invalid run ID")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Keep existing audit: {output}")
    sys.addaudithook(no_truth_reads)
    run = ROOT / ".runtime/experiments" / run_id
    prepared = read_json(run / "prepared.json")
    if prepared["state"] != "prepared" or prepared["run_id"] != run_id:
        raise ValueError("Prepared run identity mismatch")
    method_path, index_path = run / "method_config.json", run / "input_index.json"
    if digest(method_path) != prepared["method_config_sha256"] or digest(index_path) != prepared["input_manifest_sha256"]:
        raise ValueError("Frozen method or input index changed")
    method, index = read_json(method_path), read_json(index_path)
    caps = method["candidate_caps"]
    if caps != [8, 16]:
        raise ValueError("This audit is scoped to the planned cap8/cap16 experiment")
    input_frames = {case["case_id"]: case["frames"] for case in index["cases"]}
    planned = [(case, offset) for case in method["case_ids"] for offset in method["search_offsets_px"]]
    if len(planned) != 10 or len(set(planned)) != 10 or set(input_frames) != set(method["case_ids"]):
        raise ValueError("Expected the complete frozen ten-pool plan")
    expected_names = {f"{case}-search-{offset}.json" for case, offset in planned}
    if {path.name for path in (run / "pools").glob("*.json")} != expected_names:
        raise ValueError("Pool files are missing or unexpected")
    minimum_rows = method["image_hypotheses"]["minimum_rows"]
    minimum_span = method["image_hypotheses"]["minimum_y_span"]
    per_view, pool_records, retained_violations = [], [], []
    for case_id, offset in planned:
        path = run / "pools" / f"{case_id}-search-{offset}.json"
        sha = digest(path)
        pool = read_json(path)
        if (pool["case_id"], pool["search_offset_px"]) != (case_id, offset):
            raise ValueError("Pool identity mismatch")
        # cap8 finishes before cap16. Its saved reference lets this independent
        # audit verify every pool while the last cap16 association is still busy.
        reference_path = run / "associations" / f"{case_id}-search-{offset}-cap-8.json"
        reference = read_json(reference_path)
        if (reference["case_id"], reference["search_offset_px"], reference["cap"]) != (case_id, offset, 8) or reference["pool_sha256"] != sha:
            raise ValueError("Pool hash disagrees with its completed cap8 association")
        expected = {frame["view_id"]: frame for frame in input_frames[case_id]}
        actual = [frame["view_id"] for frame in pool["frames"]]
        if len(actual) != len(set(actual)) or set(actual) != set(expected):
            raise ValueError("Window/view plan mismatch")
        pool_records.append({
            "case_id": case_id, "search_offset_px": offset,
            "path": path.relative_to(ROOT).as_posix(), "sha256": sha,
            "cap8_reference_path": reference_path.relative_to(ROOT).as_posix(),
            "cap8_reference_sha256": digest(reference_path), "cap8_pool_hash_verified": True,
        })
        for frame in pool["frames"]:
            original = expected[frame["view_id"]]
            if any(frame.get(key) != value for key, value in original.items()):
                raise ValueError("Pool frame differs from the frozen input index")
            candidates = []
            for rank, hypothesis in enumerate(frame["pool"]["candidates"][:max(caps)], 1):
                matches = hypothesis["row_matches"]
                raw_rows = frame["observations"]["rows"]
                ys, seen = [], set()
                for match in matches:
                    row, candidate, _ = match
                    if isinstance(row, bool) or not isinstance(row, int) or not 0 <= row < len(raw_rows) or row in seen:
                        raise ValueError("Invalid or duplicate final row index")
                    if isinstance(candidate, bool) or not isinstance(candidate, int) or not 0 <= candidate < len(raw_rows[row]["candidates"]):
                        raise ValueError("Invalid final candidate index")
                    seen.add(row)
                    y = float(raw_rows[row]["y"])
                    if not math.isfinite(y):
                        raise ValueError("Nonfinite observed y")
                    ys.append(y)
                if hypothesis["support_rows"] != len(matches):
                    raise ValueError("Declared final support count disagrees with row_matches")
                span = max(ys) - min(ys) if ys else 0.0
                row = {
                    "case_id": case_id, "search_offset_px": offset, "view_id": frame["view_id"],
                    "rank": rank, "final_matched_rows": len(matches),
                    "final_y_min": min(ys) if ys else None, "final_y_max": max(ys) if ys else None,
                    "final_y_span_px": span,
                    "rows_below_minimum": len(matches) < minimum_rows,
                    "span_below_minimum": span < minimum_span,
                    "smallest_threshold_ratio": min(len(matches) / minimum_rows, span / minimum_span),
                }
                candidates.append(row)
                if row["rows_below_minimum"] or row["span_below_minimum"]:
                    retained_violations.append(row)
            per_view.append({
                "case_id": case_id, "search_offset_px": offset, "view_id": frame["view_id"],
                "sampled_pool_size": len(frame["pool"]["candidates"]),
                "caps": {str(cap): counts(candidates[:cap]) for cap in caps},
            })
    summaries = []
    for cap in caps:
        values = [row["caps"][str(cap)] for row in per_view]
        summaries.append({
            "cap": cap,
            **{key: sum(row[key] for row in values) for key in values[0]},
            "window_view_count": len(values),
            "affected_window_views_rows": sum(row["rows_below_minimum"] > 0 for row in values),
            "affected_window_views_span": sum(row["span_below_minimum"] > 0 for row in values),
            "affected_window_views_either": sum(row["either_below_minimum"] > 0 for row in values),
        })
    result = {
        "state": "complete", "run_id": run_id, "gt_read_during_audit": False,
        "method_sha256": digest(method_path), "input_index_sha256": digest(index_path),
        "audit_source_path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
        "audit_source_sha256": digest(Path(__file__)),
        "pool_count": len(pool_records), "thresholds": {"minimum_rows": minimum_rows, "minimum_y_span_px": minimum_span},
        "calculation": [
            "Use each saved row_matches entry to index the original raw observations; final span is max(y)-min(y).",
            "Compare final matched-row count and span to the configured minima with strict less-than; equality passes.",
            "Each cap is a prefix of the same frozen candidate pool. Count once per case/search-window/view/rank, without repeating identity offsets.",
            "Cap8 and cap16 are nested comparisons, not disjoint candidate populations. Candidate totals use actual available prefix lengths.",
            "This audits a legacy behavior: minima were checked before least-squares refitting but not checked again on final support.",
            "No truth or evaluation artifacts are read. Pool hashes are verified against completed cap8 association records.",
        ],
        "summaries": summaries, "pool_files": pool_records, "per_window_view": per_view,
        "rank_le8_violations": [row for row in retained_violations if row["rank"] <= 8],
        "worst_retained_violations": sorted(retained_violations, key=lambda row: (row["smallest_threshold_ratio"], row["final_y_span_px"], row["case_id"], row["search_offset_px"], row["view_id"], row["rank"]))[:10],
        "all_retained_violations": retained_violations,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps(summaries, indent=2))
    print("SUPPORT_AUDIT_WRITTEN", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.run_id, args.output)
