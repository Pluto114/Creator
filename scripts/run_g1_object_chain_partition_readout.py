"""Three-strategy/two-process anchor-support replay of immutable object patches."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import run_g1_object_point_patch as parent
import run_g1_object_stroke_chain_readout as control_run
import run_g1_object_stroke_chain_readout as previous_sources
from run_fixture_challenges import physical_summary

ROOT = parent.ROOT
read, write, digest, receipt, block_truth = parent.read, parent.write, parent.digest, parent.receipt, parent.block_truth
from creator_eval import single_axis_readout as single  # noqa: E402
from creator_eval.chain_partition_readout import DEFAULTS as PARTITION_DEFAULTS  # noqa: E402
from creator_eval.fixture_physical_metrics import POLICY as PHYSICAL_POLICY  # noqa: E402
from creator_eval.rgb_candidate_readout import POLICY  # noqa: E402
from creator_eval.rgb_chain_partition_axis_readout import ChainPartitionAxisReadout  # noqa: E402
from creator_eval.rgb_chain_support import DEFAULTS as CHAIN_DEFAULTS  # noqa: E402
from creator_eval.rgb_chain_support import build_context  # noqa: E402
from creator_eval.rgb_required_stroke_axis_readout import RequiredStrokeAxisReadout  # noqa: E402
from creator_eval.rgb_single_axis_readout import SingleAxisReadout  # noqa: E402
from creator_eval.rgb_stroke_chain_axis_readout import StrokeChainAxisReadout  # noqa: E402
from creator_eval.rgb_stroke_chain_support import condition_stroke_context  # noqa: E402
from creator_eval.rod_fixture_finite import canonical_hash  # noqa: E402
from creator_recon.domain.point_patch import compose, open_candidate_view  # noqa: E402

RUN_ID = "g1-object-chain-partition-readout-v1-20261009"
RUN = ROOT / ".runtime/experiments" / RUN_ID
CONFIG = ROOT / "configs/g1_object_chain_partition_readout_v1.json"
PUBLIC = ROOT / "docs/experiments/results/2026-10-09-g1-object-chain-partition-readout.json"
AUDIT = ROOT / "docs/experiments/results/2026-10-09-g1-object-chain-partition-readout-audit.json"
ADAPTERS = dict(stroke_chain_axis=StrokeChainAxisReadout, required_stroke_axis=RequiredStrokeAxisReadout,
    chain_partition_axis=ChainPartitionAxisReadout)
COUNT_KEYS = ("full_input_point_count", "supported_base_point_count", "full_input_segment_count",
              "full_curve_sample_count", "supported_curve_sample_count", "base_vote_histogram", "curve_vote_histogram")
KEYS = ("repeat", "case_id", "fraction", "variant", "reader")
OWN_SOURCES = {"scripts/run_g1_object_chain_partition_readout.py", "configs/g1_object_chain_partition_readout_v1.json", "tests/test_g1_object_chain_partition_readout.py",
    "experiments/src/creator_eval/single_axis_readout.py", "experiments/src/creator_eval/rgb_single_axis_readout.py",
    "tests/test_single_axis_readout.py", "tests/test_single_axis_readout_deepseek.py",
    "experiments/src/creator_eval/rgb_chain_support.py", "experiments/src/creator_eval/rgb_chain_axis_readout.py",
    "tests/test_rgb_chain_support.py", "tests/test_rgb_chain_support_deepseek.py",
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
    "scripts/run_fixture_chain_partition_readout.py", "configs/fixture_chain_partition_readout_v1.json",
    "tests/test_fixture_chain_partition_readout.py"}

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

def now():
    return datetime.now(timezone.utc).isoformat()


def inventory(config, repeat=None):
    repeats = config["repeats"] if repeat is None else [repeat]
    if any(type(value) is not int or value not in config["repeats"] for value in repeats):
        raise ValueError("Unknown reader-only repeat")
    return [(r, cid, fraction, variant, reader) for r in repeats for cid in config["case_ids"]
            for fraction in config["voxel_camera_span_fractions"] for variant in config["variants"] for reader in config["readers"]]


def row_key(row):
    return tuple(row[key] for key in KEYS)


def validated_config():
    config = read(CONFIG)
    expected = dict(run_id=RUN_ID, parent_run_id=parent.RUN_ID,
        parent_pre_sha256="807d7fadedc47f39e30db3cb78dfeb8934b8e706119c5e342afbccf5bd2a0af1",
        parent_inference_sha256="74bc654b83ed830c9ab53f7c9e4c194c50d7b085152c95c010eb0c9d05eea7b3",
        parent_pre_receipts=674, parent_output_receipts=62, scene_run_id=parent.parent.SCENE_ID,
        case_ids=["chair01", "aframe01"], variants=["base", *parent.controls.METHODS, "baseline", "cylinder_support"],
        voxel_camera_span_fractions=[.0025, .005], readers=["stroke_chain_axis", "required_stroke_axis", "chain_partition_axis"], repeats=[0, 1],
        expected_normal_rows=120, support_policy=POLICY, chain_policy=CHAIN_DEFAULTS, reader_overrides={},
        anchor_manifest=ANCHOR_MANIFEST, anchor_policy={"minimum_anchor_views": 2},
        stroke_policy={"minimum_anchor_views": 2, "minimum_marks_per_view": 2}, partition_policy=PARTITION_DEFAULTS,
        development_prior_scores_seen=True, blind_evaluation=False,
        input_support_intervention=True, reader_algorithm_unchanged=False, reader_only_intervention=False)
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Frozen object readout inventory/policy differs")
    return config


def source_inventory(point_pre):
    return sorted(set(previous_sources.source_inventory(point_pre)) | OWN_SOURCES)


def parent_normal(config):
    if (digest(parent.RUN / "pre.json") != config["parent_pre_sha256"]
            or digest(parent.RUN / "inference.json") != config["parent_inference_sha256"]):
        raise ValueError("Original point/patch parent receipt differs")
    before, manifest, _, normal = parent.normal_records()
    if (len(before["hashes"]) != config["parent_pre_receipts"]
            or len(normal["outputs"]) != config["parent_output_receipts"]
            or [row["case_id"] for row in normal["rows"]] != config["case_ids"]):
        raise ValueError("Complete frozen point/patch parent inventory required")
    return before, manifest, normal


def checked():
    before, config = read(RUN / "pre.json"), validated_config()
    if (before["run_id"] != RUN_ID or before["gt_read"] or before["inference_existed"]
            or before.get("reader_only_intervention") is not False or before["runtime"] != parent.runtime()
            or before.get("development_prior_scores_seen") is not True or before.get("blind_evaluation") is not False):
        raise ValueError("Invalid object readout freeze/runtime")
    check_flags(before)
    for name, sha in before["hashes"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Frozen object readout receipt changed: " + name)
    frozen = read(RUN / "evaluation-freeze.json")
    if (frozen["config"] != config or frozen["reader_defaults"] != defaults()
            or frozen["expected_normal_rows"] != len(inventory(config)) or frozen["physical_policy"] != PHYSICAL_POLICY):
        raise ValueError("Frozen readers, evaluation policy or inventory differs")
    check_flags(frozen)
    return before, config


def prepare():
    sys.addaudithook(block_truth)
    if RUN.exists() or PUBLIC.exists() or AUDIT.exists():
        raise FileExistsError("Preserve previous object readout attempt")
    config = validated_config()
    for cid in config["case_ids"]:
        anchor_input(cid)
    before, _, normal = parent_normal(config)
    hashes = {}
    freeze_controls(hashes)
    for name, sha in {**before["hashes"], **normal["outputs"]}.items():
        receipt(ROOT / name, hashes, sha)
    for path in (parent.RUN / "pre.json", parent.RUN / "inference.json"):
        receipt(path, hashes)
    names = source_inventory(before)
    if any(not (ROOT / name).is_file() for name in names):
        raise FileNotFoundError("All explicit readout sources and tests must exist before freeze")
    RUN.mkdir(parents=True)
    for name in names:
        target = RUN / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        receipt(target, hashes, receipt(ROOT / name, hashes))
    write(RUN / "evaluation-freeze.json", dict(config=config, reader_defaults=defaults(),
        expected_normal_rows=len(inventory(config)), physical_policy=PHYSICAL_POLICY,
        gt_read=False, input_support_intervention=True, reader_algorithm_unchanged=False, reader_only_intervention=False, native_geometry_unchanged=True,
        fresh_reader_processes=2, end_to_end_repeat_performed=False, independent_real_capture=False,
        reader_qualified=False, g1_passed=False, development_prior_scores_seen=True, blind_evaluation=False))
    receipt(RUN / "evaluation-freeze.json", hashes)
    write(RUN / "pre.json", dict(run_id=RUN_ID, created_at_utc=now(), hashes=hashes, source_count=len(names),
        runtime=parent.runtime(), gt_read=False, inference_existed=False, input_support_intervention=True, reader_algorithm_unchanged=False, reader_only_intervention=False,
        development_prior_scores_seen=True, blind_evaluation=False))
    checked()
    print("G1_OBJECT_CHAIN_PARTITION_READOUT_PREPARED", len(names), "sources", len(hashes), "receipts", flush=True)


def validate_pair(first, second):
    for key in (*FULL_KEYS, "call_policy", "native_segments", "native_source_state", "native_patch_id"):
        if first[key] != second[key]:
            raise ValueError("Paired readout changed original sampling/native identity: " + key)
    validate_support(first, second)


def point_rows(normal):
    return {row["case_id"]: row for row in normal["rows"]}


def native_metadata(row, variant):
    if variant == "base":
        return dict(native_segments=[], native_source_state="unchanged_base", native_patch_id=None,
                    native_outcome="no_added_geometry", native_reason=[])
    patch = next(item for item in row["patches"] if item["method"] == variant)
    manifest = read(parent.RUN / row["case_id"] / f"bundle/patch-{variant}/manifest.json")
    if manifest["content_id"] != patch["patch_id"]:
        raise ValueError("Original native patch manifest identity differs")
    return dict(native_segments=patch["native_segments"], native_source_state=patch["source_state"],
                native_patch_id=patch["patch_id"], native_outcome=patch["outcome"], native_reason=manifest["metadata"]["unresolved"])


def infer_repeat(repeat):
    sys.addaudithook(block_truth)
    _, config = checked()
    if type(repeat) is not int or repeat not in config["repeats"]:
        raise ValueError("Unknown reader-only repeat")
    record_dir, repeat_path = RUN / f"records-r{repeat}", RUN / f"repeat-{repeat}.json"
    if record_dir.exists() or repeat_path.exists():
        raise FileExistsError("Preserve previous object reader repeat")
    _, manifest, original = parent_normal(config)
    originals = point_rows(original)
    record_dir.mkdir()
    rows, associations, started = [], [], time.perf_counter()
    for case in manifest["cases"]:
        cid, folder, bound = case["case_id"], parent.RUN / case["case_id"], originals[case["case_id"]]
        raw = read(ROOT / case["parent_record"])
        cameras = read(folder / "rods.json")["cameras"]
        support = read(folder / "support.json")
        if (canonical_hash(cameras) != bound["camera_export_sha256"]
                or canonical_hash(cameras) != support["camera_sha256"]
                or canonical_hash(raw["frames"]) != support["raw_frame_sha256"]):
            raise ValueError("Exact exported camera/raw RGB support identity differs")
        context = build_context(raw["frames"], cameras)
        anchors, anchor_metadata = anchor_input(cid)
        stroke_context = condition_stroke_context(context, raw["frames"], cameras, anchors)
        contexts = {name: stroke_context for name in CONTEXT_READERS}
        for name in CONTEXT_READERS:
            associations.append(save_context(record_dir, cid, contexts[name], name))
        views, base = context["views"], compose(folder / "bundle/base")
        if (base["base_snapshot_id"] != bound["snapshot_id"] or len(base["points"]) != bound["full_point_count"]
                or base["world_frame_id"] != "fixture-conditioned-prediction:" + bound["prediction_sha256"]
                or base["length_unit"] != "meter"):
            raise ValueError("Complete original base snapshot identity differs")
        adapters = {name: ADAPTERS[name](base["points"], contexts[name] if name in CONTEXT_READERS else views,
                    config["support_policy"]) for name in config["readers"]}
        first = SingleAxisReadout(base["points"], views, config["support_policy"])
        for name in CONTEXT_READERS:
            validate_adapters(first, adapters[name])
        validate_partition_adapter(adapters["required_stroke_axis"], adapters["chain_partition_axis"])
        if (len(first.points) != bound["supported_base_point_count"]
                or np.bincount(first.votes, minlength=len(views)+1).tolist() != support["vote_histogram"]):
            raise ValueError("Original complete base RGB mask/votes changed")
        span = parent.camera_span(np.asarray([camera["world_to_camera_cv"] for camera in cameras]))
        if span != bound["camera_span_m"]:
            raise ValueError("Original exported camera span changed")
        candidates, native = {"base": base}, {}
        for variant in config["variants"]:
            native[variant] = native_metadata(bound, variant)
            if variant != "base":
                candidates[variant] = open_candidate_view(folder / f"bundle/{variant}-enabled.json")
                segments = np.asarray(native[variant]["native_segments"], float).reshape(-1, 2, 3)
                parent.validate_candidate(base, candidates[variant], segments, enabled=True)
                parent.validate_candidate(base, open_candidate_view(folder / f"bundle/{variant}-withdrawn.json"), segments, enabled=False)
        for fraction in config["voxel_camera_span_fractions"]:
            call = dict(voxel_size=span*fraction, origin=[0., 0., 0.])
            for variant in config["variants"]:
                pair = []
                for name in config["readers"]:
                    call_started = time.perf_counter()
                    output = adapters[name](candidates[variant]["segments"], call)
                    output.update(anchor_metadata)
                    output.update(graph_flags(name))
                    if name == "chain_partition_axis":
                        output.update(additional_anchor_consumer=name, additional_annotation_cost=0)
                    output.update(gt_read=False, repeat=repeat, case_id=cid, fraction=fraction, variant=variant, reader=name,
                        reader_elapsed_seconds=time.perf_counter()-call_started, input_support_intervention=True, reader_algorithm_unchanged=False, reader_only_intervention=False,
                        native_state="complete", native_metric_scope="added_segments_only_not_raw_point_quality",
                        base_snapshot_id=base["base_snapshot_id"], source_support_sha256=bound["support_sha256"],
                        raw_row_states=[dict(view_id=view["view_id"], **view["row_states"]) for view in views], **native[variant])
                    output.update(graph_flags(name))
                    if name in CONTEXT_READERS and output["chain_context_sha256"] != contexts[name]["sha256"]:
                        raise ValueError("Chain adapter context identity differs")
                    if output["effective_policy"] != {**defaults()[name], **call}:
                        raise ValueError("Unchanged reader defaults/call policy differ")
                    if name == "chain_partition_axis":
                        validate_partition_pair(pair[1], output)
                    pair.append(output)
                    if len(pair) > 1:
                        validate_pair(pair[0], pair[-1])
                    path = record_dir / f"{cid}-{fraction}-{variant}-{name}.json"
                    write(path, output)
                    rows.append(dict(repeat=repeat, case_id=cid, fraction=fraction, variant=variant, reader=name,
                        path=path.relative_to(RUN).as_posix(), sha256=digest(path)))
                    print("G1_OBJECT_CHAIN_PARTITION_READOUT", repeat, cid, fraction, variant, name, output["state"], len(output["segments"]), flush=True)
    checked()
    if [row_key(row) for row in rows] != inventory(config, repeat):
        raise ValueError("Complete sixty-row reader repeat required")
    write(repeat_path, dict(run_id=RUN_ID, repeat=repeat, state="complete", gt_read=False, rows=rows,
        pre_sha256=digest(RUN / "pre.json"), associations=associations,
        elapsed_seconds=time.perf_counter()-started, **FLAGS))


def repeat_normal(config, repeat):
    normal = read(RUN / f"repeat-{repeat}.json")
    if (normal["run_id"] != RUN_ID or normal["repeat"] != repeat or normal["state"] != "complete" or normal["gt_read"]
            or normal["pre_sha256"] != digest(RUN / "pre.json") or normal.get("reader_only_intervention") is not False
            or [row_key(row) for row in normal["rows"]] != inventory(config, repeat)):
        raise ValueError("Complete sixty-row reader repeat required")
    check_flags(normal)
    verify_contexts(config, repeat, normal["associations"])
    return normal


def infer():
    sys.addaudithook(block_truth)
    _, config = checked()
    if (RUN / "inference.json").exists():
        raise FileExistsError("Preserve completed object readout")
    write(RUN / "inference-started.json", dict(run_id=RUN_ID, created_at_utc=now(), repeats=config["repeats"], **FLAGS))
    rows, repeats, started = [], [], time.perf_counter()
    for repeat in config["repeats"]:
        subprocess.run([sys.executable, "-B", str(ROOT / "scripts/run_g1_object_chain_partition_readout.py"),
                        "infer-repeat", str(repeat)], cwd=ROOT, check=True, timeout=3600)
        normal = repeat_normal(config, repeat)
        rows.extend(normal["rows"])
        path = RUN / f"repeat-{repeat}.json"
        repeats.append(dict(repeat=repeat, path=path.relative_to(RUN).as_posix(), sha256=digest(path),
                            elapsed_seconds=normal["elapsed_seconds"]))
    checked()
    write(RUN / "inference.json", dict(run_id=RUN_ID, state="complete", gt_read=False, rows=rows, repeats=repeats,
        pre_sha256=digest(RUN / "pre.json"), input_support_intervention=True, reader_algorithm_unchanged=False, reader_only_intervention=False, fresh_reader_processes=2,
        end_to_end_repeat_performed=False, elapsed_seconds=time.perf_counter()-started,
        development_prior_scores_seen=True, blind_evaluation=False))
    verified_normal(config)


def verified_normal(config):
    normal = read(RUN / "inference.json")
    if (normal["run_id"] != RUN_ID or normal["state"] != "complete" or normal["gt_read"]
            or normal["pre_sha256"] != digest(RUN / "pre.json") or normal.get("reader_only_intervention") is not False
            or normal["fresh_reader_processes"] != 2 or normal["end_to_end_repeat_performed"]
            or normal.get("development_prior_scores_seen") is not True or normal.get("blind_evaluation") is not False
            or [row_key(row) for row in normal["rows"]] != inventory(config)):
        raise ValueError("All 120 frozen normal rows required before any truth access")
    check_flags(normal)
    if [entry["repeat"] for entry in normal["repeats"]] != config["repeats"]:
        raise ValueError("Both fresh reader-process receipts required")
    repeat_rows, associations = [], {}
    for entry in normal["repeats"]:
        if (entry["path"] != f"repeat-{entry['repeat']}.json"
                or digest(RUN / entry["path"]) != entry["sha256"]):
            raise ValueError("Reader-repeat receipt changed")
        repeated = repeat_normal(config, entry["repeat"])
        repeat_rows.extend(repeated["rows"])
        associations[entry["repeat"]] = verify_contexts(config, entry["repeat"], repeated["associations"])
    if repeat_rows != normal["rows"]:
        raise ValueError("Fresh reader-process row pairing differs")
    _, manifest, original = parent_normal(config)
    originals, graphs, pairs, required_graphs = point_rows(original), [], {}, {}
    for row in normal["rows"]:
        expected_path = f"records-r{row['repeat']}/{row['case_id']}-{row['fraction']}-{row['variant']}-{row['reader']}.json"
        if row["path"] != expected_path or digest(RUN / row["path"]) != row["sha256"]:
            raise ValueError("Normal readout graph changed")
        graph, bound = read(RUN / row["path"]), originals[row["case_id"]]
        check_flags(graph)
        validate_anchor_input(graph)
        if row["reader"] in CONTEXT_READERS and graph["chain_context_sha256"] != associations[row["repeat"]][row["case_id"]][row["reader"]]["chain_context_sha256"]:
            raise ValueError("Normal graph/association identity differs")
        expected_native = native_metadata(bound, row["variant"])
        call = dict(voxel_size=bound["camera_span_m"]*row["fraction"], origin=[0., 0., 0.])
        if (graph["gt_read"] or any(graph[key] != row[key] for key in KEYS)
                or graph["call_policy"] != call or graph["effective_policy"] != {**defaults()[row["reader"]], **call}
                or graph["base_snapshot_id"] != bound["snapshot_id"]
                or graph["full_input_point_count"] != bound["full_point_count"]
                or (row["reader"] == "single_axis" and graph["supported_base_point_count"] != bound["supported_base_point_count"])
                or graph["source_support_sha256"] != bound["support_sha256"]
                or any(graph[key] != value for key, value in expected_native.items())):
            raise ValueError("Normal reader/complete base/native identity differs")
        key = row_key(row)[:-1]
        if row["reader"] == config["readers"][0]:
            pairs[key] = graph
        else:
            validate_pair(pairs[key], graph)
        if row["reader"] == "required_stroke_axis":
            required_graphs[key] = graph
        elif row["reader"] == "chain_partition_axis":
            validate_partition_pair(required_graphs[key], graph)
        graphs.append(graph)
    verify_old_controls(normal["rows"], graphs)
    return normal, graphs, manifest, original


def repeat_projection(graph):
    keys = ("state", "resolution_state", "reason", "segments", "native_segments", "native_source_state",
        "native_patch_id", "native_outcome", "native_reason", "call_policy", "effective_policy", "raw_row_states",
        "base_snapshot_id", "source_support_sha256", *COUNT_KEYS, *CHAIN_META_KEYS, *ANCHOR_META_KEYS, *PARTITION_META_KEYS)
    return {key: graph.get(key) for key in keys}


def score_geometry(state, segments, truth, gaps):
    if state != "complete":
        return None, None
    try:
        return physical_summary(segments, truth, gaps), None
    except ValueError as exc:
        return None, str(exc)


def evaluation_payload():
    before, config = checked()
    normal, graphs, manifest, original = verified_normal(config)
    # The full 120-row normal and complete parent artifact inventories are
    # verified above before this run accesses truth. Prior development scores
    # were already seen; this is not blind evaluation.
    scene = ROOT / ".runtime/experiments" / config["scene_run_id"]
    truth_path = ROOT / "data/eval_gt" / config["scene_run_id"] / "manifest.json"
    if digest(truth_path) != read(scene / "prepared.json")["truth_sha256"]:
        raise ValueError("Frozen new-object truth identity differs")
    truth = read(truth_path)
    if (truth["input_sha256"] != digest(ROOT / "data/inputs" / config["scene_run_id"] / "manifest.json")
            or [case["case_id"] for case in truth["cases"]] != config["case_ids"]):
        raise ValueError("New-object truth/normal input pairing differs")
    truths = {}
    for source, actual in zip(manifest["cases"], truth["cases"]):
        if [frame["rgb_sha256"] for frame in source["frames"]] != [frame["rgb_sha256"] for frame in actual["frames"]]:
            raise ValueError("Object truth RGB identity/order differs")
        declared, target = actual["declared"], actual["declared"]["target"]
        segments = (target["segments"] if "segments" in target else [target["endpoints"]]) if target["present"] else []
        gaps = [declared["gap_segment"]] if "gap_segment" in declared else []
        truths[source["case_id"]] = segments, gaps
    rows, native_rows, projections, native_seen = [], [], {}, set()
    for entry, graph in zip(normal["rows"], graphs):
        truth_segments, gaps = truths[entry["case_id"]]
        score, error = score_geometry(graph["state"], graph["segments"], truth_segments, gaps)
        key = entry["case_id"], entry["fraction"], entry["variant"], entry["reader"]
        projection = repeat_projection(graph)
        repeated = None if entry["repeat"] == 0 else projection == projections[key]
        if entry["repeat"] == 0:
            projections[key] = projection
        rows.append(dict(**entry, state=graph["state"], resolution_state=graph.get("resolution_state"),
            reason=graph.get("reason"), output_segment_count=len(graph["segments"]), physical=score, scoring_error=error,
            native_source_state=graph["native_source_state"], native_outcome=graph["native_outcome"],
            native_patch_id=graph["native_patch_id"], native_segment_count=len(graph["native_segments"]),
            reader_elapsed_seconds=graph["reader_elapsed_seconds"], physical_equal_to_repeat_zero=repeated,
            **{name: graph[name] for name in COUNT_KEYS}, **support_metadata(graph)))
        native_key = entry["case_id"], entry["variant"]
        if native_key not in native_seen:
            native_seen.add(native_key)
            native_score, native_error = score_geometry("complete", graph["native_segments"], truth_segments, gaps)
            native_rows.append(dict(case_id=entry["case_id"], variant=entry["variant"],
                source_state=graph["native_source_state"], outcome=graph["native_outcome"], reasons=graph["native_reason"],
                patch_id=graph["native_patch_id"], segments=graph["native_segments"], physical=native_score,
                scoring_error=native_error, measurement_scope=graph["native_metric_scope"],
                is_point_quality_measurement=False, scored_without_readout_clipping=True))
    if [(row["case_id"], row["variant"]) for row in native_rows] != [(cid, variant) for cid in config["case_ids"] for variant in config["variants"]]:
        raise ValueError("Complete ten native arms required")
    return dict(run_id=RUN_ID, state="evaluated", pre_sha256=digest(RUN / "pre.json"),
        inference_sha256=digest(RUN / "inference.json"), truth_sha256=digest(truth_path),
        unchanged_pre_receipts=len(before["hashes"]), elapsed_seconds=normal["elapsed_seconds"],
        rows=rows, native_rows=native_rows, reader_repeats=normal["repeats"],
        reader_repeat_pairs=60, reader_repeat_equal=sum(row["physical_equal_to_repeat_zero"] is True for row in rows),
        source_parent_run_id=parent.RUN_ID, source_parent_inference_sha256=config["parent_inference_sha256"],
        source_parent_elapsed_seconds=original["elapsed_seconds"], parent_integration=original["rows"],
        parent_output_receipts=len(original["outputs"]), physical_policy=PHYSICAL_POLICY,
        measurement_scope="declared_central_rod_local_RGB_volume_not_other_members_surfaces_or_whole_object_quality",
        gpu_memory_scope="parent torch peak allocated for the process, not whole-GPU memory utilization",
        native_geometry_unchanged=True, point_inputs_unchanged=True, no_patch_refit=True,
        synthetic_development=True, independent_real_capture=False, end_to_end_repeat_performed=False,
        development_prior_scores_seen=True, blind_evaluation=False,
        input_support_intervention=True, reader_algorithm_unchanged=False, reader_only_intervention=False,
        target_identity_confirmed=False, alignment_performed=False, reader_qualified=False, g1_passed=False, qualification=config["qualification"])


def evaluate():
    if PUBLIC.exists() or (RUN / "evaluation.json").exists():
        raise FileExistsError("Preserve previous object readout evaluation")
    result = evaluation_payload()
    write(RUN / "evaluation.json", result)
    write(PUBLIC, result)
    print("G1_OBJECT_CHAIN_PARTITION_READOUT_EVALUATED", len(result["rows"]), "common", len(result["native_rows"]), "native", flush=True)


def post():
    if AUDIT.exists() or (RUN / "post.json").exists():
        raise FileExistsError("Preserve previous object readout audit")
    public = read(PUBLIC)
    if public != evaluation_payload() or PUBLIC.read_bytes() != (RUN / "evaluation.json").read_bytes():
        raise ValueError("Object readout evaluation replay differs")
    result = dict(run_id=RUN_ID, state="passed", created_at_utc=now(), public_sha256=digest(PUBLIC),
        unchanged_pre_receipts=public["unchanged_pre_receipts"], exact_evaluation_replay=True,
        reader_repeat_pairs=public["reader_repeat_pairs"], reader_repeat_equal=public["reader_repeat_equal"],
        same_complete_base=True, chain_mask_subset_raw=True, same_full_sampling_counts=True, native_geometry_unchanged=True,
        point_inputs_unchanged=True, no_patch_refit=True, reader_only_repeats=False, support_intervention_repeats=True,
        input_support_intervention=True, reader_algorithm_unchanged=False, reader_only_intervention=False,
        end_to_end_repeat_performed=False, independent_real_capture=False, reader_qualified=False, g1_passed=False,
        development_prior_scores_seen=True, blind_evaluation=False)
    write(RUN / "post.json", result)
    write(AUDIT, result)
    print("G1_OBJECT_CHAIN_PARTITION_READOUT_POST_PASSED", flush=True)


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
