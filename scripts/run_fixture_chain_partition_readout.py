"""Fixed stroke/required controls versus partition readout: 180 fixture rows."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time

import numpy as np
import run_fixture_naive_controls as previous
import run_fixture_stroke_chain_readout as control_run
import run_fixture_stroke_chain_readout as previous_sources
from run_fixture_challenges import digest, now, physical_summary, read, receipt, write

ROOT, parent, analytic = previous.ROOT, previous.parent, previous.analytic
from creator_eval import single_axis_readout as single  # noqa: E402
from creator_eval.chain_partition_readout import DEFAULTS as PARTITION_DEFAULTS  # noqa: E402
from creator_eval.rgb_candidate_readout import POLICY  # noqa: E402
from creator_eval.rgb_chain_partition_axis_readout import ChainPartitionAxisReadout  # noqa: E402
from creator_eval.rgb_chain_support import DEFAULTS as CHAIN_DEFAULTS  # noqa: E402
from creator_eval.rgb_chain_support import build_context  # noqa: E402
from creator_eval.rgb_required_stroke_axis_readout import RequiredStrokeAxisReadout  # noqa: E402
from creator_eval.rgb_single_axis_readout import SingleAxisReadout  # noqa: E402
from creator_eval.rgb_stroke_chain_axis_readout import StrokeChainAxisReadout  # noqa: E402
from creator_eval.rgb_stroke_chain_support import condition_stroke_context  # noqa: E402
from creator_eval.rod_fixture_finite import canonical_hash  # noqa: E402
from run_g1_object_chain_partition_readout import repeat_projection  # noqa: E402

SCOPE = "Fixed stroke/required controls versus same-chain actual-geometry partition readout; complete original GT, no target-identity claim"
from creator_recon.domain.point_patch import compose, open_candidate_view  # noqa: E402

RUN_ID = "fixture-chain-partition-readout-v1-20261009"
RUN = ROOT / ".runtime/experiments" / RUN_ID
CONFIG = ROOT / "configs/fixture_chain_partition_readout_v1.json"
PUBLIC = ROOT / "docs/experiments/results/2026-10-09-fixture-chain-partition-readout.json"
AUDIT = ROOT / "docs/experiments/results/2026-10-09-fixture-chain-partition-readout-audit.json"
block_truth = previous.block_truth
KEYS = ("repeat", "case_id", "fraction", "variant", "reader")
ADAPTERS = dict(stroke_chain_axis=StrokeChainAxisReadout, required_stroke_axis=RequiredStrokeAxisReadout,
    chain_partition_axis=ChainPartitionAxisReadout)
COUNT_KEYS = ("full_input_point_count", "supported_base_point_count", "full_input_segment_count",
              "full_curve_sample_count", "supported_curve_sample_count", "base_vote_histogram", "curve_vote_histogram")
NATIVE_KEYS = ("native_segments", "native_state", "native_metric_scope", "control", "raw_row_states")

FLAGS = dict(input_support_intervention=True, reader_algorithm_unchanged=False,
    reader_only_intervention=False, development_prior_scores_seen=True, blind_evaluation=False)
FULL_KEYS = ("full_input_point_count", "full_input_segment_count", "full_curve_sample_count")
CHAIN_META_KEYS = ("raw_base_vote_histogram", "raw_curve_vote_histogram",
    "base_chain_mask_subset_raw", "curve_chain_mask_subset_raw", "chain_context_sha256",
    "chain_association", "target_identity_confirmed")



ANCHOR_MANIFEST = "data/inputs/rgb-foreground-strokes-v1-20261009/manifest.json"
ANCHOR_META_KEYS = ("anchor_input_sha256", "anchor_input_metadata", "anchor_claims_sha256",
                    "support_domain_kind", "anchor_view_count", "anchor_point_count", "stroke_interpolation_performed",
                    "required_support_view_ids", "required_support_views_applied", "required_support_rule")
CONTEXT_READERS = ("stroke_chain_axis", "required_stroke_axis", "chain_partition_axis")

CONTROL_READERS = ("stroke_chain_axis", "required_stroke_axis")
PARTITION_META_KEYS = ("global_union_fit_performed", "partition_algorithm", "geometry_merge_kind",
    "readouts", "fitted_support_groups", "positive_groups", "raw_group_segment_count",
    "retained_groups", "suppressed_aliases", "alias_point_segment_checks", "alias_policy",
    "resolution_distinct_geometries_retained", "axis_search_exhaustive",
    "additional_anchor_consumer", "additional_annotation_cost")


def graph_flags(name):
    return {**FLAGS, "reader_algorithm_unchanged": name in CONTROL_READERS}


def defaults():
    return dict(stroke_chain_axis=single.DEFAULTS, required_stroke_axis=single.DEFAULTS,
        chain_partition_axis={"axis": single.DEFAULTS, "partition": PARTITION_DEFAULTS})


def validate_partition_pair(required, partition):
    validate_alias_audit(partition)
    for key in COUNT_KEYS:
        if required[key] != partition[key]:
            raise ValueError("Partition must preserve exact required-view unique union counts: " + key)
    if partition.get("global_union_fit_performed") is not False:
        raise ValueError("Partition must not fit the global union")



def validate_alias_audit(graph):
    if graph["state"] != "complete":
        return
    readouts, retained, aliases = graph["readouts"], graph["retained_groups"], graph["suppressed_aliases"]
    kept = set(retained)
    if len(kept) != len(retained) or any(type(i) is not int or not 0 <= i < len(readouts) for i in kept):
        raise ValueError("Invalid retained partition groups")
    suppressed = set()
    for alias in aliases:
        group, peer = alias["group"], alias["retained_group"]
        bound, maximum = alias["continuous_distance_upper_bound_m"], alias["maximum_sample_distance_m"]
        shared, size = alias["shared_fraction_of_smaller_set"], graph["call_policy"]["voxel_size"]
        if (type(group) is not int or not 0 <= group < len(readouts) or group in kept or group in suppressed
                or peer not in kept or not np.isfinite([bound, maximum, shared]).all()
                or not 0 <= maximum <= bound <= size * PARTITION_DEFAULTS["alias_distance_voxels"]
                or not PARTITION_DEFAULTS["minimum_shared_support_fraction"] <= shared <= 1.
                or not np.isclose(bound, maximum + size * PARTITION_DEFAULTS["alias_sampling_step_voxels"] / 2,
                                  rtol=1e-12, atol=0.)):
            raise ValueError("Invalid continuous alias certificate or transitive suppressed peer")
        suppressed.add(group)

def validate_partition_adapter(required, partition):
    if (not np.array_equal(required.mask, partition.mask)
            or not np.array_equal(required.points, partition.points)):
        raise ValueError("Partition and required adapter must use the exact same unique base union")



def anchor_input(cid):
    manifest = read(ROOT / ANCHOR_MANIFEST)
    cases = manifest["cases"]
    if [case["case_id"] for case in cases] != ["r01", "r02", "r03", "chair01", "aframe01"]:
        raise ValueError("Complete five-case RGB anchor input required")
    case = next(case for case in cases if case["case_id"] == cid)
    anchors = case["anchors"]
    if not isinstance(anchors, list) or len(anchors) < 2:
        raise ValueError("At least two explicit RGB anchor views required")
    metadata = dict(manifest={k: v for k, v in manifest.items() if k != "cases"},
        case={k: v for k, v in case.items() if k != "anchors"},
        anchor_count=len(anchors), anchor_view_count=len({a["view_id"] for a in anchors}),
        source="codex_visual_rgb_annotation_not_human_or_GT",
        anchors_shared_across_all_method_arms=True,
        anchors_used_by_readers=["stroke_chain_axis", "required_stroke_axis"],
        source_claim_not_identity_proof=True)
    return anchors, dict(anchor_input_sha256=digest(ROOT / ANCHOR_MANIFEST), anchor_input_metadata=metadata)


def validate_anchor_input(graph):
    _, metadata = anchor_input(graph["case_id"])
    if any(graph.get(key) != value for key, value in metadata.items()):
        raise ValueError("Frozen common anchor provenance/cost differs")


def freeze_controls(hashes):
    before, config = control_run.checked()
    normal = control_run.verified_normal(config)[0]
    for name, sha in before["hashes"].items():
        receipt(ROOT / name, hashes, sha)
    for path in (control_run.RUN / "pre.json", control_run.RUN / "inference.json"):
        receipt(path, hashes)
    for row in normal["rows"]:
        receipt(control_run.RUN / row["path"], hashes, row["sha256"])
    for repeat in normal["repeats"]:
        receipt(control_run.RUN / repeat["path"], hashes, repeat["sha256"])
        record = read(control_run.RUN / repeat["path"])
        for entry in record["associations"]:
            receipt(control_run.RUN / entry["path"], hashes, entry["sha256"])


def verify_old_controls(rows, graphs):
    _, config = control_run.checked()
    old = control_run.verified_normal(config)
    controls = old[1]
    if controls and isinstance(controls[0], tuple):
        controls = [pair[0] for pair in controls]
    by_key = {row_key(row): graph for row, graph in zip(old[0]["rows"], controls) if row["reader"] in CONTROL_READERS}
    matched = 0
    for row, graph in zip(rows, graphs):
        if row["reader"] in CONTROL_READERS:
            if control_run.repeat_projection(graph) != control_run.repeat_projection(by_key[row_key(row)]):
                raise ValueError("October 9 stroke/required control changed")
            matched += 1
    if matched != len(by_key):
        raise ValueError("Every October 9 stroke/required control required")


def check_flags(record):
    expected = graph_flags(record["reader"]) if "reader" in record else FLAGS
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("Support intervention/development identity differs")


def support_metadata(graph):
    return {key: graph.get(key) for key in (*CHAIN_META_KEYS, *ANCHOR_META_KEYS, *PARTITION_META_KEYS)}


def validate_adapters(raw, chain):
    if (not np.array_equal(raw.votes, chain.raw_votes)
            or not np.array_equal(raw.votes, chain.votes)
            or chain.mask.shape != raw.mask.shape or np.any(chain.mask & ~raw.mask)
            or len(chain.points) != int(chain.mask.sum())):
        raise ValueError("Chain base mask must be a subset of the original raw RGB mask")


def validate_support(raw, chain):
    for key in FULL_KEYS:
        if raw[key] != chain[key]:
            raise ValueError("Full input/sampling counts changed: " + key)
    for prefix in ("base", "curve"):
        if chain["raw_"+prefix+"_vote_histogram"] != raw[prefix+"_vote_histogram"]:
            raise ValueError("Raw RGB votes changed")
        if chain[prefix+"_vote_histogram"] != raw[prefix+"_vote_histogram"]:
            raise ValueError("Original vote histograms must stay raw")
        if chain[prefix+"_chain_mask_subset_raw"] is not True:
            raise ValueError("Chain mask subset contract failed")
        if not 0 <= chain["supported_"+prefix+("_point_count" if prefix == "base" else "_sample_count")] <= raw["supported_"+prefix+("_point_count" if prefix == "base" else "_sample_count")]:
            raise ValueError("Chain support cannot expand the raw RGB domain")
    if chain.get("target_identity_confirmed") is not False:
        raise ValueError("Chains are hypotheses, not confirmed target identity")


def save_context(record_dir, cid, context, name):
    path = record_dir / f"{cid}-{name}-association.json"
    write(path, context["association"])
    return dict(case_id=cid, reader=name, path=path.relative_to(RUN).as_posix(), sha256=digest(path),
                chain_context_sha256=context["sha256"])


def verify_contexts(config, repeat, entries):
    expected_pairs = [(cid, name) for cid in config["case_ids"] for name in CONTEXT_READERS]
    if [(entry["case_id"], entry["reader"]) for entry in entries] != expected_pairs:
        raise ValueError("Complete per-case all-chain and anchor association receipts required")
    by_case = {}
    for entry in entries:
        expected = f"records-r{repeat}/{entry['case_id']}-{entry['reader']}-association.json"
        if entry["path"] != expected or digest(RUN / expected) != entry["sha256"]:
            raise ValueError("Frozen chain association changed")
        if canonical_hash(read(RUN / expected)) != entry["chain_context_sha256"]:
            raise ValueError("Association content/context SHA differs")
        by_case.setdefault(entry["case_id"], {})[entry["reader"]] = entry
    return by_case

def inventory(config, repeat=None):
    repeats = config["repeats"] if repeat is None else [repeat]
    if any(type(value) is not int or value not in config["repeats"] for value in repeats):
        raise ValueError("Unknown support intervention repeat")
    return [(r, cid, fraction, variant, name) for r in repeats for cid in config["case_ids"]
        for fraction in config["voxel_camera_span_fractions"] for variant in config["variants"] for name in config["readers"]]


def row_key(row):
    return tuple(row[key] for key in KEYS)


def validated_config():
    config = read(CONFIG)
    required = dict(run_id=RUN_ID, parent_run_id=previous.RUN_ID, case_ids=["r01", "r02", "r03"],
        voxel_camera_span_fractions=[.0025, .005],
        variants=["base", *previous.controls.METHODS, "baseline", "cylinder_support"],
        readers=["stroke_chain_axis", "required_stroke_axis", "chain_partition_axis"], repeats=[0, 1], expected_normal_rows=180, support_policy=POLICY, chain_policy=CHAIN_DEFAULTS, reader_overrides={},
        anchor_manifest=ANCHOR_MANIFEST, anchor_policy={"minimum_anchor_views": 2},
        stroke_policy={"minimum_anchor_views": 2, "minimum_marks_per_view": 2}, partition_policy=PARTITION_DEFAULTS,
        development_prior_scores_seen=True, blind_evaluation=False,
        input_support_intervention=True, reader_algorithm_unchanged=False, reader_only_intervention=False)
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError("Frozen RGB-chain support inventory or policy differs")
    return config


def source_inventory():
    return sorted(set(previous_sources.source_inventory()) | {
        "scripts/run_fixture_chain_partition_readout.py", "configs/fixture_chain_partition_readout_v1.json",
        "tests/test_fixture_chain_partition_readout.py", "experiments/src/creator_eval/rgb_chain_support.py",
        "experiments/src/creator_eval/rgb_chain_axis_readout.py", "tests/test_rgb_chain_support.py",
        "tests/test_rgb_chain_support_deepseek.py",
        "experiments/src/creator_eval/rgb_anchored_chain_support.py",
        "experiments/src/creator_eval/rgb_stroke_chain_support.py", "tests/test_rgb_stroke_chain_support.py",
        "tests/test_rgb_stroke_chain_axis_readout.py",
        "experiments/src/creator_eval/rgb_required_view_support.py",
        "experiments/src/creator_eval/rgb_required_stroke_axis_readout.py",
        "tests/test_rgb_required_view_support.py", "tests/test_rgb_required_stroke_axis_readout.py",
        "experiments/src/creator_eval/chain_segment_union.py", "experiments/src/creator_eval/chain_partition_readout.py",
        "experiments/src/creator_eval/rgb_chain_partition_axis_readout.py",
        "tests/test_chain_partition_readout.py", "tests/test_chain_segment_union_deepseek.py",
        "experiments/src/creator_eval/rgb_stroke_chain_axis_readout.py",
        "tests/test_rgb_anchored_chain_support.py", "tests/test_rgb_anchor_states_deepseek.py",
        "data/inputs/rgb-foreground-strokes-v1-20261009/manifest.json",
        "data/inputs/rgb-foreground-strokes-v1-20261009/annotation-audit.json",
        "scripts/run_g1_object_chain_partition_readout.py", "configs/g1_object_chain_partition_readout_v1.json",
        "tests/test_g1_object_chain_partition_readout.py",
        "experiments/src/creator_eval/single_axis_readout.py",
        "experiments/src/creator_eval/rgb_single_axis_readout.py", "tests/test_single_axis_readout.py"})


def checked():
    before, config = read(RUN / "pre.json"), validated_config()
    if (before["run_id"] != RUN_ID or before["gt_read"] or before["inference_existed"]
            or before.get("reader_only_intervention") is not False or before["runtime"] != analytic.runtime()
            or before.get("development_prior_scores_seen") is not True or before.get("blind_evaluation") is not False):
        raise ValueError("Invalid rgb-chain development freeze or changed runtime")
    check_flags(before)
    for name, sha in before["hashes"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Frozen rgb-chain receipt changed: " + name)
    frozen = read(RUN / "evaluation-freeze.json")
    if (frozen["config"] != config or frozen["reader_defaults"] != defaults()
            or frozen["expected_normal_rows"] != len(inventory(config))
            or frozen.get("development_prior_scores_seen") is not True or frozen.get("blind_evaluation") is not False):
        raise ValueError("Frozen rgb-chain policy or inventory differs")
    check_flags(frozen)
    return before, config


def parent_normal(config):
    before, old_config = previous.checked()
    if previous.inventory(old_config) != list(dict.fromkeys(row[:-1] for row in inventory(config))):
        raise ValueError("All 180 original repeat/case/scale/variant pairs required")
    normal, pairs = previous.verified_normal(old_config)
    graphs = {previous.row_key(entry): graph for entry, (graph, _) in zip(normal["rows"], pairs)}
    return before, normal, graphs


def prepare():
    sys.addaudithook(block_truth)
    if RUN.exists() or PUBLIC.exists() or AUDIT.exists():
        raise FileExistsError("Preserve every prior rgb-chain attempt")
    config = validated_config()
    for cid in config["case_ids"]:
        anchor_input(cid)
    before, normal, _ = parent_normal(config)
    hashes = {}
    freeze_controls(hashes)
    for name, sha in {**before["hashes"], **normal["outputs"]}.items():
        receipt(ROOT / name, hashes, sha)
    for path in (previous.RUN / "pre.json", previous.RUN / "inference.json"):
        receipt(path, hashes)
    for row in normal["rows"]:
        receipt(previous.RUN / row["path"], hashes, row["sha256"])
    names = source_inventory()
    if any(not (ROOT / name).is_file() for name in names):
        raise FileNotFoundError("All explicit reader and independent-test sources must exist before freeze")
    RUN.mkdir(parents=True)
    for name in names:
        target = RUN / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        receipt(target, hashes, receipt(ROOT / name, hashes))
    write(RUN / "evaluation-freeze.json", dict(config=config, expected_normal_rows=len(inventory(config)),
        reader_defaults=defaults(), physical_policy=read(previous.RUN / "evaluation-freeze.json")["physical_policy"],
        gt_read=False, input_support_intervention=True, reader_algorithm_unchanged=False, reader_only_intervention=False, native_geometry_unchanged=True,
        parent_inference_sha256=digest(previous.RUN / "inference.json"),
        development_prior_scores_seen=True, blind_evaluation=False, measurement_scope=SCOPE,
        fresh_process_repeats=True, independent_objects=False, reader_qualified=False, g1_passed=False))
    receipt(RUN / "evaluation-freeze.json", hashes)
    write(RUN / "pre.json", dict(run_id=RUN_ID, created_at_utc=now(), gt_read=False, inference_existed=False,
        input_support_intervention=True, reader_algorithm_unchanged=False, reader_only_intervention=False, runtime=analytic.runtime(), development_prior_scores_seen=True, blind_evaluation=False,
        source_count=len(names), hashes=hashes))
    checked()
    print("FIXTURE_CHAIN_PARTITION_PREPARED", len(names), "sources", len(hashes), "receipts", flush=True)


def candidate_paths(cid, variant, repeat):
    if cid not in ("r01", "r02", "r03") or repeat not in (0, 1):
        raise ValueError("Unknown frozen case or repeat")
    base = previous.RUN / cid / "bundle/base"
    if variant == "base":
        return base, None, None
    if variant in previous.controls.METHODS:
        bundle, name = previous.RUN / cid / "bundle", f"{variant}-r{repeat}"
    elif variant in ("baseline", "cylinder_support"):
        bundle, name = parent.RUN / cid / "bundle", variant
    else:
        raise ValueError("Unknown frozen variant")
    return base, bundle / f"{name}-enabled.json", bundle / f"{name}-withdrawn.json"


def validate_pair(graph, old):
    immutable = COUNT_KEYS if graph["reader"] == "single_axis" else FULL_KEYS
    for key in (*immutable, *NATIVE_KEYS, "call_policy"):
        if graph[key] != old[key]:
            raise ValueError("Complete-input pairing changed: " + key)
    if graph["reader"] in CONTEXT_READERS:
        validate_support(old, graph)
    if graph["effective_policy"] != {**defaults()[graph["reader"]], **graph["call_policy"]}:
        raise ValueError("RGB-chain effective policy differs")


def infer_repeat(repeat):
    sys.addaudithook(block_truth)
    _, config = checked()
    if type(repeat) is not int or repeat not in config["repeats"]:
        raise ValueError("Unknown rgb-chain repeat")
    inventory(config, repeat)
    record_dir, repeat_path = RUN / f"records-r{repeat}", RUN / f"repeat-{repeat}.json"
    if record_dir.exists() or repeat_path.exists():
        raise FileExistsError("Preserve every prior rgb-chain repeat")
    _, old_normal, old_graphs = parent_normal(config)
    old_rows = {previous.row_key(row): row for row in old_normal["rows"]}
    _, manifest, _ = parent.checked()
    point_rows = {row["case_id"]: row for row in read(parent.RUN / "inference.json")["rows"]}
    record_dir.mkdir()
    rows, associations, started = [], [], time.perf_counter()
    for case in manifest["cases"]:
        cid = case["case_id"]
        raw = read(ROOT / case["parent_record"])
        cameras = read(parent.RUN / cid / "rods.json")["cameras"]
        context = build_context(raw["frames"], cameras)
        anchors, anchor_metadata = anchor_input(cid)
        stroke_context = condition_stroke_context(context, raw["frames"], cameras, anchors)
        contexts = {name: stroke_context for name in CONTEXT_READERS}
        for name in CONTEXT_READERS:
            associations.append(save_context(record_dir, cid, contexts[name], name))
        views = context["views"]
        base = compose(candidate_paths(cid, "base", repeat)[0])
        original = point_rows[cid]
        if (base["base_snapshot_id"] != original["snapshot_id"] or len(base["points"]) != original["full_point_count"]
                or base["length_unit"] != "meter"
                or base["world_frame_id"] != "fixture-conditioned-prediction:" + original["prediction_sha256"]):
            raise ValueError("Original full base snapshot/prediction frame differs")
        candidates = {"base": base}
        for variant in config["variants"][1:]:
            _, enabled, withdrawn = candidate_paths(cid, variant, repeat)
            candidates[variant] = open_candidate_view(enabled)
            previous.validate_candidate(base, candidates[variant])
            previous.validate_candidate(base, open_candidate_view(withdrawn), withdrawn=True)
        adapters = {name: ADAPTERS[name](base["points"], contexts[name] if name in CONTEXT_READERS else views,
                    config["support_policy"]) for name in config["readers"]}
        raw_adapter = SingleAxisReadout(base["points"], views, config["support_policy"])
        for name in CONTEXT_READERS:
            validate_adapters(raw_adapter, adapters[name])
        validate_partition_adapter(adapters["required_stroke_axis"], adapters["chain_partition_axis"])
        for fraction in config["voxel_camera_span_fractions"]:
            for variant in config["variants"]:
                key = repeat, cid, fraction, variant
                old, reference = old_graphs[key], old_rows[key]
                native = np.asarray(old["native_segments"], float).reshape(-1, 2, 3)
                if not np.array_equal(candidates[variant]["segments"], native):
                    raise ValueError("Reopened original geometry differs from its paired native record")
                call = old["call_policy"]
                if call != dict(voxel_size=previous.camera_span(cameras)*fraction, origin=[0., 0., 0.]):
                    raise ValueError("Original camera-span scale/origin changed")
                outputs = {}
                for name in config["readers"]:
                    call_started = time.perf_counter()
                    output = adapters[name](candidates[variant]["segments"], call)
                    output.update(anchor_metadata)
                    output.update(graph_flags(name))
                    if name == "chain_partition_axis":
                        output.update(additional_anchor_consumer=name, additional_annotation_cost=0)
                    output.update({name: old[name] for name in NATIVE_KEYS})
                    output.update(gt_read=False, repeat=repeat, case_id=cid, fraction=fraction, variant=variant, reader=name,
                        reader_elapsed_seconds=time.perf_counter()-call_started, input_support_intervention=True, reader_algorithm_unchanged=False, reader_only_intervention=False,
                        development_prior_scores_seen=True, blind_evaluation=False)
                    if name in CONTEXT_READERS and output["chain_context_sha256"] != contexts[name]["sha256"]:
                        raise ValueError("Chain adapter context identity differs")
                    output.update(graph_flags(name))
                    if name == "chain_partition_axis":
                        validate_partition_pair(outputs["required_stroke_axis"], output)
                    outputs[name] = output
                    validate_pair(output, old)
                    path = record_dir / f"{cid}-{fraction}-{variant}-{name}.json"
                    write(path, output)
                    rows.append(dict(repeat=repeat, case_id=cid, fraction=fraction, variant=variant, reader=name,
                        path=path.relative_to(RUN).as_posix(), sha256=digest(path),
                        previous_path=reference["path"], previous_sha256=reference["sha256"]))
                    print("FIXTURE_CHAIN_PARTITION", repeat, cid, fraction, variant, name, output["state"], len(output["segments"]), flush=True)
    checked()
    if [row_key(row) for row in rows] != inventory(config, repeat):
        raise ValueError("Incomplete rgb-chain repeat inventory")
    write(repeat_path, dict(run_id=RUN_ID, repeat=repeat, state="complete", gt_read=False, rows=rows,
        pre_sha256=digest(RUN / "pre.json"), associations=associations,
        elapsed_seconds=time.perf_counter()-started, **FLAGS))


def repeat_normal(config, repeat):
    normal = read(RUN / f"repeat-{repeat}.json")
    if (normal["run_id"] != RUN_ID or normal["repeat"] != repeat or normal["state"] != "complete"
            or normal["gt_read"] or normal["pre_sha256"] != digest(RUN / "pre.json")
            or normal.get("development_prior_scores_seen") is not True or normal.get("blind_evaluation") is not False
            or [row_key(row) for row in normal["rows"]] != inventory(config, repeat)):
        raise ValueError("Complete rgb-chain repeat inventory required")
    check_flags(normal)
    verify_contexts(config, repeat, normal["associations"])
    return normal


def infer():
    sys.addaudithook(block_truth)
    _, config = checked()
    started = time.perf_counter()
    write(RUN / "inference-started.json", dict(run_id=RUN_ID, created_at_utc=now(), repeats=config["repeats"], **FLAGS))
    rows, repeats = [], []
    for repeat in config["repeats"]:
        subprocess.run([sys.executable, "-B", str(ROOT / "scripts/run_fixture_chain_partition_readout.py"),
            "infer-repeat", str(repeat)], cwd=ROOT, check=True, timeout=1800)
        normal = repeat_normal(config, repeat)
        rows.extend(normal["rows"])
        path = RUN / f"repeat-{repeat}.json"
        repeats.append(dict(repeat=repeat, path=path.relative_to(RUN).as_posix(), sha256=digest(path),
            elapsed_seconds=normal["elapsed_seconds"]))
    checked()
    write(RUN / "inference.json", dict(run_id=RUN_ID, state="complete", gt_read=False, rows=rows,
        repeats=repeats, fresh_process_repeats=True, input_support_intervention=True, reader_algorithm_unchanged=False, reader_only_intervention=False,
        pre_sha256=digest(RUN / "pre.json"), elapsed_seconds=time.perf_counter()-started,
        development_prior_scores_seen=True, blind_evaluation=False))
    verified_normal(config)


def verified_normal(config):
    """All 180 fresh graphs and their original bundles are verified before GT."""
    normal = read(RUN / "inference.json")
    if (normal["run_id"] != RUN_ID or normal["state"] != "complete" or normal["gt_read"]
            or normal["pre_sha256"] != digest(RUN / "pre.json") or not normal["fresh_process_repeats"]
            or normal.get("reader_only_intervention") is not False
            or normal.get("development_prior_scores_seen") is not True or normal.get("blind_evaluation") is not False
            or [row_key(row) for row in normal["rows"]] != inventory(config)):
        raise ValueError("All 180 rgb-chain normal rows required before truth")
    if [entry["repeat"] for entry in normal["repeats"]] != config["repeats"]:
        raise ValueError("Missing fresh-process repeat receipts")
    check_flags(normal)
    repeated, associations = [], {}
    for entry in normal["repeats"]:
        path = RUN / f"repeat-{entry['repeat']}.json"
        if entry["path"] != path.relative_to(RUN).as_posix() or digest(path) != entry["sha256"]:
            raise ValueError("RGB-chain repeat receipt changed")
        repeated_normal = repeat_normal(config, entry["repeat"])
        repeated.extend(repeated_normal["rows"])
        associations[entry["repeat"]] = verify_contexts(config, entry["repeat"], repeated_normal["associations"])
    if repeated != normal["rows"]:
        raise ValueError("RGB-chain repeat row references differ")
    _, old_normal, old_graphs = parent_normal(config)
    graphs, required_graphs = [], {}
    old_rows = {previous.row_key(row): row for row in old_normal["rows"]}
    for row in normal["rows"]:
        reference = old_rows[row_key(row)[:-1]]
        if row_key(row)[:-1] != previous.row_key(reference) or row["previous_path"] != reference["path"] or row["previous_sha256"] != reference["sha256"]:
            raise ValueError("Exact old/new normal pairing differs")
        expected = f"records-r{row['repeat']}/{row['case_id']}-{row['fraction']}-{row['variant']}-{row['reader']}.json"
        if row["path"] != expected or digest(RUN / row["path"]) != row["sha256"]:
            raise ValueError("RGB-chain normal graph changed")
        graph, old = read(RUN / row["path"]), old_graphs[row_key(row)[:-1]]
        if (graph["gt_read"] or graph.get("reader_only_intervention") is not False
                or graph.get("development_prior_scores_seen") is not True or graph.get("blind_evaluation") is not False
                or any(graph[key] != row[key] for key in KEYS)):
            raise ValueError("RGB-chain normal identity differs")
        check_flags(graph)
        validate_anchor_input(graph)
        if row["reader"] in CONTEXT_READERS and graph["chain_context_sha256"] != associations[row["repeat"]][row["case_id"]][row["reader"]]["chain_context_sha256"]:
            raise ValueError("Normal graph/association identity differs")
        validate_pair(graph, old)
        if row["reader"] == "required_stroke_axis":
            required_graphs[row_key(row)[:-1]] = graph
        elif row["reader"] == "chain_partition_axis":
            validate_partition_pair(required_graphs[row_key(row)[:-1]], graph)
        graphs.append((graph, old))
    verify_old_controls(normal["rows"], [pair[0] for pair in graphs])
    return normal, graphs


def evaluation_payload():
    before, config = checked()
    normal, graphs = verified_normal(config)
    scene_id = read(parent.CONFIG)["scene_run_id"]
    truth_path = ROOT / "data/eval_gt" / scene_id / "manifest.json"
    if digest(truth_path) != read(ROOT / ".runtime/experiments" / scene_id / "prepared.json")["truth_sha256"]:
        raise ValueError("Original truth identity differs")
    truth = {case["case_id"]: case for case in read(truth_path)["cases"]}
    rows, projections = [], {}
    for entry, (graph, old) in zip(normal["rows"], graphs):
        declared = truth[entry["case_id"]]["declared"]
        target = declared["target"]
        segments = target.get("segments", [target.get("endpoints")]) if target["present"] else []
        gaps = [declared["gap_segment"]] if "gap_segment" in declared else []
        scores, errors = [], []
        for state, values in ((graph["state"], graph["segments"]), (old["state"], old["segments"]),
                              (graph["native_state"], graph["native_segments"]), (old["native_state"], old["native_segments"])):
            score, error = None, None
            if state == "complete":
                try:
                    score = physical_summary(values, segments, gaps)
                except ValueError as exc:
                    error = str(exc)
            scores.append(score)
            errors.append(error)
        if scores[2] != scores[3] or errors[2] != errors[3]:
            raise ValueError("Unchanged native geometry physical recomputation differs")
        key = entry["case_id"], entry["fraction"], entry["variant"], entry["reader"]
        projection = repeat_projection(graph)
        repeated = None if entry["repeat"] == 0 else projection == projections[key]
        if entry["repeat"] == 0:
            projections[key] = projection
        rows.append(dict(**entry, state=graph["state"], reason=graph.get("reason"),
            output_segment_count=len(graph["segments"]), previous_state=old["state"],
            previous_output_segment_count=len(old["segments"]), native_segment_count=len(graph["native_segments"]),
            physical=scores[0], previous_physical=scores[1], native_physical=scores[2],
            previous_native_physical=scores[3], native_physical_unchanged=True,
            scoring_error=errors[0], previous_scoring_error=errors[1], native_scoring_error=errors[2],
            native_metric_scope=graph["native_metric_scope"], control=graph["control"],
            reader_elapsed_seconds=graph["reader_elapsed_seconds"], physical_equal_to_repeat_zero=repeated,
            **{key: graph[key] for key in COUNT_KEYS}, **support_metadata(graph)))
    return dict(run_id=RUN_ID, state="evaluated", pre_sha256=digest(RUN / "pre.json"),
        inference_sha256=digest(RUN / "inference.json"), parent_inference_sha256=digest(previous.RUN / "inference.json"),
        truth_sha256=digest(truth_path), unchanged_pre_receipts=len(before["hashes"]), elapsed_seconds=normal["elapsed_seconds"],
        rows=rows, base_rows=[row for row in rows if row["variant"] == "base"],
        fresh_process_repeats=normal["repeats"], physical_repeat_pairs=90,
        physical_repeat_equal=sum(row["physical_equal_to_repeat_zero"] is True for row in rows),
        native_physical_recomputed_from_original_geometry=True, native_unchanged_rows=len(rows),
        input_support_intervention=True, reader_algorithm_unchanged=False, reader_only_intervention=False, reader_qualified=False, independent_objects=False, g1_passed=False,
        physical_policy=read(RUN / "evaluation-freeze.json")["physical_policy"], measurement_scope=SCOPE,
        development_prior_scores_seen=True, blind_evaluation=False, alignment_performed=False, qualification=config["qualification"])


def evaluate():
    if PUBLIC.exists() or (RUN / "evaluation.json").exists():
        raise FileExistsError("Preserve prior rgb-chain evaluation")
    result = evaluation_payload()
    write(RUN / "evaluation.json", result)
    write(PUBLIC, result)
    print("FIXTURE_CHAIN_PARTITION_EVALUATED", len(result["rows"]), flush=True)


def post():
    if AUDIT.exists() or (RUN / "post.json").exists():
        raise FileExistsError("Preserve prior rgb-chain post audit")
    public = read(PUBLIC)
    if public != evaluation_payload() or PUBLIC.read_bytes() != (RUN / "evaluation.json").read_bytes():
        raise ValueError("RGB-chain exact evaluation replay differs")
    result = dict(run_id=RUN_ID, state="passed", created_at_utc=now(), public_sha256=digest(PUBLIC),
        unchanged_pre_receipts=public["unchanged_pre_receipts"], exact_evaluation_replay=True,
        input_support_intervention=True, reader_algorithm_unchanged=False, reader_only_intervention=False, original_bundles_unchanged=True, full_measurement_counts_unchanged=True, chain_mask_subset_raw=True,
        native_unchanged_rows=public["native_unchanged_rows"], physical_repeat_pairs=90,
        physical_repeat_equal=public["physical_repeat_equal"], reader_qualified=False, g1_passed=False,
        independent_objects=False, measurement_scope=SCOPE,
        development_prior_scores_seen=True, blind_evaluation=False)
    write(RUN / "post.json", result)
    write(AUDIT, result)
    print("FIXTURE_CHAIN_PARTITION_POST_PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "infer", "infer-repeat", "evaluate", "post"))
    parser.add_argument("repeat", type=int, nargs="?")
    args = parser.parse_args()
    from environment_paths import require_project_environment

    require_project_environment(ROOT)
    if args.stage == "infer-repeat":
        infer_repeat(args.repeat)
    elif args.repeat is not None:
        parser.error("repeat is only valid for infer-repeat")
    else:
        globals()[args.stage]()
