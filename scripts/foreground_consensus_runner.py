"""Direct-parent, normal-before-GT runner shared by two fixed development suites."""

from __future__ import annotations

import argparse
import copy
import shutil
import subprocess
import sys
import time

import direct_parent_index as direct
import numpy as np
import run_fixture_chain_partition_readout as fixture
import run_g1_object_chain_partition_readout as objects

ROOT = fixture.ROOT
read, write, digest = fixture.read, fixture.write, fixture.digest
from creator_eval.rod_fixture_finite import json_ready  # noqa: E402

KEYS = fixture.KEYS
READERS = ["stroke_chain_axis", "chain_partition_axis", "foreground_consensus_axis"]
CONTROLS = READERS[:2]
NEW = READERS[-1]
CASES = {"fixture": ["r01", "r02", "r03"], "g1_object": ["chair01", "aframe01"]}
FLAGS = dict(input_support_intervention=True, reader_algorithm_unchanged=False,
    reader_only_intervention=False, development_prior_scores_seen=True, blind_evaluation=False)
NEW_SOURCES = {
    "scripts/direct_parent_index.py", "tests/test_direct_parent_index.py",
    "scripts/foreground_consensus_runner.py", "tests/test_foreground_consensus_runner.py",
    "scripts/run_fixture_foreground_consensus_readout.py", "scripts/run_g1_object_foreground_consensus_readout.py",
    "configs/fixture_foreground_consensus_readout_v1.json", "configs/g1_object_foreground_consensus_readout_v1.json",
    "experiments/src/creator_eval/foreground_consensus_readout.py",
    "experiments/src/creator_eval/foreground_support.py",
    "experiments/src/creator_eval/rgb_segment_cells.py", "tests/test_rgb_segment_cells.py",
    "experiments/src/creator_eval/rgb_foreground_consensus_axis_readout.py",
    "tests/test_foreground_consensus_readout.py",
    "tests/test_foreground_consensus_adversarial.py", "tests/test_foreground_support_deepseek.py",
}
VOLATILE_CONTROL = {"fit_record_path", "fit_record_sha256", "fit_elapsed_seconds", "enabled_path", "withdrawn_path"}


def physical_projection(graph, *, cross_repeat=False):
    """All graph keys participate; only explicitly named execution provenance differs."""
    result = json_ready(copy.deepcopy(graph))
    result.pop("reader_elapsed_seconds", None)
    if cross_repeat:
        result.pop("repeat", None)
        if isinstance(result.get("control"), dict):
            for key in VOLATILE_CONTROL:
                result["control"].pop(key, None)
    return result


def row_key(row):
    return tuple(row[key] for key in KEYS)


class Runner:
    def __init__(self, kind):
        if kind not in CASES:
            raise ValueError("Unknown fixed suite")
        self.kind = kind
        self.old = fixture if kind == "fixture" else objects
        self.run_id = kind.replace("_", "-") + "-foreground-consensus-readout-v1-20261012"
        self.run = ROOT / ".runtime/experiments" / self.run_id
        self.config_path = ROOT / "configs" / (kind + "_foreground_consensus_readout_v1.json")
        stem = "2026-10-12-" + kind.replace("_", "-") + "-foreground-consensus-readout"
        self.public = ROOT / "docs/experiments/results" / (stem + ".json")
        self.audit = self.public.with_name(stem + "-audit.json")
        self.point = self.old.parent

    def config(self):
        config = read(self.config_path)
        expected = dict(run_id=self.run_id, parent_run_id=self.old.RUN_ID, case_ids=CASES[self.kind],
            variants=["base", "depth_tls_single_span", "depth_ransac_single_span", "baseline", "cylinder_support"],
            readers=READERS, repeats=[0, 1], voxel_camera_span_fractions=[.0025, .005],
            expected_normal_rows=len(CASES[self.kind])*60, support_policy=self.old.POLICY,
            reader_overrides={}, anchor_manifest=self.old.ANCHOR_MANIFEST, **FLAGS)
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError("Complete fixed inventory, policies and non-blind status required")
        return config

    def inventory(self, config, repeat=None):
        repeats = config["repeats"] if repeat is None else [repeat]
        if any(type(value) is not int or value not in config["repeats"] for value in repeats):
            raise ValueError("Unknown normal repeat")
        return [(r, cid, fraction, variant, name) for r in repeats for cid in config["case_ids"]
            for fraction in config["voxel_camera_span_fractions"] for variant in config["variants"] for name in READERS]

    def sources(self, before):
        prefix = self.old.RUN.relative_to(ROOT).as_posix() + "/source_snapshot/"
        inherited = {name[len(prefix):] for name in before["hashes"] if name.startswith(prefix)}
        if len(inherited) != before["source_count"]:
            raise ValueError("Previous closed source inventory differs")
        return sorted(inherited | NEW_SOURCES)

    def defaults(self):
        from creator_eval import foreground_consensus_readout as consensus
        from creator_eval import foreground_support as support
        from creator_eval import rgb_segment_cells as cells
        from creator_eval import single_axis_readout as axis
        return {**{key: self.old.defaults()[key] for key in CONTROLS},
            NEW: dict(axis=axis.DEFAULTS, consensus=consensus.DEFAULTS,
                support=support.DEFAULTS, seed_group_limits=consensus.SEED_GROUP_LIMITS, rgb_cells=cells.DEFAULTS)}

    def candidates(self, cid, variant, repeat):
        if self.kind == "fixture":
            return fixture.candidate_paths(cid, variant, repeat)
        folder = self.point.RUN / cid / "bundle"
        return folder / "base", (folder / (variant + "-enabled.json") if variant != "base" else None), \
            (folder / (variant + "-withdrawn.json") if variant != "base" else None)

    def _input_roles(self, config, old_normal, sources):
        roles = {}
        def add(role, path):
            roles.setdefault(role, set()).add(path.relative_to(ROOT).as_posix())
        def bundle(folder, role):
            add(role, folder / "manifest.json")
            body = read(folder / "manifest.json")
            for ref in body["arrays"].values():
                if ref["file"] != ref["file"].replace("/", "").replace("\\", ""):
                    raise ValueError("Nonlocal bundle array reference")
                add(role, folder / ref["file"])
        def view(path):
            add("candidate_view", path)
            body = read(path)
            for role, ref in body["references"].items():
                if ref is not None:
                    if not ref or "/" in ref or "\\" in ref or ref in (".", ".."):
                        raise ValueError("Nonlocal candidate view reference")
                    bundle(path.parent / ref, "base_snapshot" if role == "snapshot" else "patch")
        for name in sources:
            add("source", ROOT / name)
        for path in (self.point.INPUTS / "manifest.json", ROOT / self.old.ANCHOR_MANIFEST):
            add("input_manifest", path)
        for path in (self.point.CONFIG, self.old.CONFIG, self.config_path):
            add("source", path)
        for path in (self.point.RUN / "pre.json", self.old.RUN / "pre.json"):
            add("parent_pre", path)
        add("parent_normal", self.point.RUN / "inference.json")
        add("control_normal", self.old.RUN / "inference.json")
        for entry in old_normal["rows"]:
            if entry["reader"] in CONTROLS:
                add("control_graph", self.old.RUN / entry["path"])
        for entry in old_normal["repeats"]:
            path = self.old.RUN / entry["path"]
            add("control_repeat", path)
            for assoc in read(path)["associations"]:
                add("control_association", self.old.RUN / assoc["path"])
        manifest = read(self.point.INPUTS / "manifest.json")
        if [case["case_id"] for case in manifest["cases"]] != config["case_ids"]:
            raise ValueError("Complete original case manifest required")
        for case in manifest["cases"]:
            cid = case["case_id"]
            add("raw_record", ROOT / case["parent_record"])
            for frame in case["frames"]:
                add("rgb", ROOT / frame["rgb"])
            add("cameras", self.point.RUN / cid / "rods.json")
            if self.kind == "g1_object":
                add("support", self.point.RUN / cid / "support.json")
            for repeat in config["repeats"]:
                for variant in config["variants"]:
                    base, enabled, withdrawn = self.candidates(cid, variant, repeat)
                    bundle(base, "base_snapshot")
                    if enabled is not None:
                        view(enabled)
                        view(withdrawn)
        scene_id = read(self.point.CONFIG)["scene_run_id"]
        add("evaluation_binding", ROOT / ".runtime/experiments" / scene_id / "prepared.json")
        add("input_manifest", ROOT / "data/inputs" / scene_id / "manifest.json")
        return {role: sorted(names) for role, names in roles.items()}

    def prepare(self):
        sys.addaudithook(self.old.block_truth)
        if self.run.exists() or self.public.exists() or self.audit.exists():
            raise FileExistsError("Preserve every previous foreground consensus attempt")
        started, config = time.perf_counter(), self.config()
        # The only call into recursive producer verification in the new runner.
        before, old_config = self.old.checked()
        old_normal = self.old.verified_normal(old_config)[0]
        ancestors = dict(before["hashes"])
        for path in (self.old.RUN / "pre.json", self.old.RUN / "inference.json"):
            ancestors[path.relative_to(ROOT).as_posix()] = digest(path)
        for row in old_normal["rows"]:
            ancestors[(self.old.RUN / row["path"]).relative_to(ROOT).as_posix()] = row["sha256"]
        for row in old_normal["repeats"]:
            path = self.old.RUN / row["path"]
            ancestors[path.relative_to(ROOT).as_posix()] = row["sha256"]
            for assoc in read(path)["associations"]:
                ancestors[(self.old.RUN / assoc["path"]).relative_to(ROOT).as_posix()] = assoc["sha256"]
        sources = self.sources(before)
        index = direct.build_index(ROOT, run_id=self.run_id, parent_run_id=self.old.RUN_ID,
            roles=self._input_roles(config, old_normal, sources), ancestor_hashes=ancestors)
        self.run.mkdir(parents=True)
        snapshots = {}
        for name in sources:
            target = self.run / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
            snapshots[target.relative_to(ROOT).as_posix()] = index["hashes"][name]
        write(self.run / "direct-parent-index.json", index)
        write(self.run / "evaluation-freeze.json", dict(config=config, defaults=self.defaults(),
            physical_policy=read(self.old.RUN / "evaluation-freeze.json")["physical_policy"],
            expected_normal_rows=len(self.inventory(config)), gt_read=False, **FLAGS))
        hashes = {**index["hashes"], **snapshots}
        for name in ("direct-parent-index.json", "evaluation-freeze.json"):
            hashes[(self.run / name).relative_to(ROOT).as_posix()] = digest(self.run / name)
        write(self.run / "pre.json", dict(run_id=self.run_id, parent_run_id=self.old.RUN_ID,
            gt_read=False, inference_existed=False, created_at_utc=fixture.now(),
            direct_index_sha256=digest(self.run / "direct-parent-index.json"), hashes=hashes,
            source_count=len(sources), ancestor_receipts=len(ancestors), runtime=self.runtime(),
            elapsed_seconds=time.perf_counter()-started, **FLAGS))
        self.checked()
        print("FOREGROUND_PREPARED", self.kind, len(sources), "sources", len(index["hashes"]), "direct", len(ancestors), "ancestor", flush=True)

    def runtime(self):
        return fixture.analytic.runtime() if self.kind == "fixture" else self.point.runtime()

    def checked(self, *, ancestors=False):
        before, config = read(self.run / "pre.json"), self.config()
        if (before["run_id"] != self.run_id or before["parent_run_id"] != self.old.RUN_ID
                or before["gt_read"] is not False or before["inference_existed"] is not False
                or before["runtime"] != self.runtime() or any(before.get(k) != v for k, v in FLAGS.items())):
            raise ValueError("Frozen runtime, identity or development status changed")
        direct.validate_hashes(before["hashes"])
        index = direct.read_index(ROOT, self.run / "direct-parent-index.json", before["direct_index_sha256"],
            run_id=self.run_id, parent_run_id=self.old.RUN_ID, ancestors=ancestors)
        if any(before["hashes"].get(name) != sha for name, sha in index["hashes"].items()):
            raise ValueError("Own pre and typed direct-input SHA inventories disagree")
        direct.verify_hashes(ROOT, {name: sha for name, sha in before["hashes"].items() if name not in index["hashes"]})
        frozen = read(self.run / "evaluation-freeze.json")
        if frozen["config"] != config or frozen["defaults"] != self.defaults() or frozen["expected_normal_rows"] != len(self.inventory(config)):
            raise ValueError("Frozen defaults or full inventory differ")
        return before, config, index

    def old_graphs(self):
        normal = read(self.old.RUN / "inference.json")
        rows = [row for row in normal["rows"] if row["reader"] in CONTROLS]
        return {row_key(row): (row, read(self.old.RUN / row["path"])) for row in rows}

    def infer_repeat(self, repeat):
        from creator_eval.rgb_foreground_consensus_axis_readout import (
            ForegroundConsensusAxisReadout,
        )
        sys.addaudithook(self.old.block_truth)
        _, config, index = self.checked()
        self.inventory(config, repeat)
        folder, repeat_path = self.run / f"records-r{repeat}", self.run / f"repeat-{repeat}.json"
        if folder.exists() or repeat_path.exists():
            raise FileExistsError("Preserve every normal attempt")
        guard = direct.DirectReadGuard(ROOT, index, self.run,
            runtime_directories=[ROOT / "backends/da3/.venv", ROOT / "reconstruction/.venv", ROOT / ".venv-tools"])
        sys.addaudithook(guard)
        started, rows, associations = time.perf_counter(), [], []
        old_graphs = self.old_graphs()
        manifest = read(self.point.INPUTS / "manifest.json")
        point_rows = {row["case_id"]: row for row in read(self.point.RUN / "inference.json")["rows"]}
        folder.mkdir()
        for case in manifest["cases"]:
            cid, bound = case["case_id"], point_rows[case["case_id"]]
            raw = read(ROOT / case["parent_record"])
            cameras = read(self.point.RUN / cid / "rods.json")["cameras"]
            span = fixture.previous.camera_span(cameras)
            context = self.old.build_context(raw["frames"], cameras)
            anchors, anchor_meta = self.old.anchor_input(cid)
            context = self.old.condition_stroke_context(context, raw["frames"], cameras, anchors)
            assoc_path = folder / (cid + "-association.json")
            write(assoc_path, context["association"])
            associations.append(dict(case_id=cid, path=assoc_path.relative_to(self.run).as_posix(),
                sha256=digest(assoc_path), chain_context_sha256=context["sha256"]))
            base = self.old.compose(self.candidates(cid, "base", repeat)[0])
            if (base["base_snapshot_id"] != bound["snapshot_id"] or len(base["points"]) != bound["full_point_count"]
                    or base["length_unit"] != "meter" or base["world_frame_id"] != "fixture-conditioned-prediction:" + bound["prediction_sha256"]):
                raise ValueError("Full original point snapshot identity differs")
            adapters = {"stroke_chain_axis": self.old.StrokeChainAxisReadout(base["points"], context, config["support_policy"]),
                "chain_partition_axis": self.old.ChainPartitionAxisReadout(base["points"], context, config["support_policy"]),
                NEW: ForegroundConsensusAxisReadout(base["points"], context, config["support_policy"])}
            if not np.array_equal(adapters[NEW].mask, adapters["stroke_chain_axis"].mask):
                raise ValueError("New reader must preserve exact stroke-positive base support domain")
            candidates = {"base": base}
            for variant in config["variants"][1:]:
                _, reference = old_graphs[(repeat, cid, config["voxel_camera_span_fractions"][0], variant, "stroke_chain_axis")]
                native = np.asarray(reference["native_segments"], float).reshape(-1, 2, 3)
                _, enabled, withdrawn = self.candidates(cid, variant, repeat)
                candidate = self.old.open_candidate_view(enabled)
                disabled = self.old.open_candidate_view(withdrawn)
                if self.kind == "fixture":
                    fixture.previous.validate_candidate(base, candidate)
                    fixture.previous.validate_candidate(base, disabled, withdrawn=True)
                else:
                    self.point.validate_candidate(base, candidate, native, enabled=True)
                    self.point.validate_candidate(base, disabled, native, enabled=False)
                candidates[variant] = candidate
            for fraction in config["voxel_camera_span_fractions"]:
                for variant in config["variants"]:
                    key = repeat, cid, fraction, variant
                    _, reference = old_graphs[(*key, "stroke_chain_axis")]
                    if reference["call_policy"] != dict(voxel_size=span*fraction, origin=[0., 0., 0.]):
                        raise ValueError("Original camera-span scale/origin changed")
                    native = np.asarray(reference["native_segments"], float).reshape(-1, 2, 3)
                    candidate = candidates[variant]
                    if not np.array_equal(candidate["segments"], native):
                        raise ValueError("Original native geometry changed")
                    metadata_keys = [k for k in reference if k.startswith("native_")]
                    metadata_keys += [k for k in ("control", "raw_row_states", "base_snapshot_id", "source_support_sha256") if k in reference]
                    metadata = {k: reference[k] for k in metadata_keys}
                    outputs = {}
                    for name in READERS:
                        call_started = time.perf_counter()
                        graph = adapters[name](candidate["segments"], reference["call_policy"])
                        if graph["effective_policy"] != {**self.defaults()[name], **reference["call_policy"]}:
                            raise ValueError("Frozen effective reader policy changed")
                        graph.update(anchor_meta)
                        graph.update(metadata)
                        graph.update(**FLAGS, gt_read=False, repeat=repeat, case_id=cid, fraction=fraction,
                            variant=variant, reader=name, reader_elapsed_seconds=time.perf_counter()-call_started)
                        graph["reader_algorithm_unchanged"] = name == "stroke_chain_axis"
                        if name != "stroke_chain_axis":
                            graph.update(additional_anchor_consumer=name, additional_annotation_cost=0)
                        if name in CONTROLS:
                            if physical_projection(graph) != physical_projection(old_graphs[(*key, name)][1]):
                                raise ValueError("Frozen control full graph changed: " + name)
                        else:
                            self.validate_new_pair(outputs["stroke_chain_axis"], graph)
                        outputs[name] = graph
                        path = folder / f"{cid}-{fraction}-{variant}-{name}.json"
                        write(path, graph)
                        rows.append(dict(zip(KEYS, (*key, name)), path=path.relative_to(self.run).as_posix(), sha256=digest(path)))
                        print("FOREGROUND_NORMAL", self.kind, *key, name, graph["state"], len(graph["segments"]), flush=True)
        guard.active = False
        self.checked()
        if [row_key(row) for row in rows] != self.inventory(config, repeat):
            raise ValueError("Incomplete normal repeat")
        write(repeat_path, dict(run_id=self.run_id, repeat=repeat, state="complete", gt_read=False,
            pre_sha256=digest(self.run / "pre.json"), rows=rows, associations=associations,
            opened_direct_inputs=sorted(guard.opened), elapsed_seconds=time.perf_counter()-started, **FLAGS))

    @staticmethod
    def validate_new_pair(stroke, new):
        for key in (*fixture.COUNT_KEYS, "raw_base_vote_histogram", "raw_curve_vote_histogram", "call_policy",
                    "native_segments", "chain_context_sha256"):
            if stroke[key] != new[key]:
                raise ValueError("Fixed stroke full-input/native pairing changed: " + key)
        if new["target_identity_confirmed"] is not False:
            raise ValueError("Positive support is not a target-identity proof")

    def infer(self):
        sys.addaudithook(self.old.block_truth)
        _, config, _ = self.checked()
        if (self.run / "inference-started.json").exists():
            raise FileExistsError("Preserve earlier inference attempt")
        started, rows, repeats = time.perf_counter(), [], []
        write(self.run / "inference-started.json", dict(run_id=self.run_id, gt_read=False, **FLAGS))
        for repeat in config["repeats"]:
            subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / ("run_" + self.kind + "_foreground_consensus_readout.py")),
                "infer-repeat", str(repeat)], cwd=ROOT, check=True, timeout=3600)
            path = self.run / f"repeat-{repeat}.json"
            normal = read(path)
            rows.extend(normal["rows"])
            repeats.append(dict(repeat=repeat, path=path.name, sha256=digest(path), elapsed_seconds=normal["elapsed_seconds"]))
        self.checked()
        write(self.run / "inference.json", dict(run_id=self.run_id, state="complete", gt_read=False,
            pre_sha256=digest(self.run / "pre.json"), rows=rows, repeats=repeats,
            fresh_reader_processes=2, end_to_end_repeat_performed=False,
            elapsed_seconds=time.perf_counter()-started, **FLAGS))
        self.normal(config)

    def normal(self, config):
        normal = read(self.run / "inference.json")
        if (normal["run_id"] != self.run_id or normal["state"] != "complete" or normal["gt_read"] is not False
                or normal["pre_sha256"] != digest(self.run / "pre.json") or normal["fresh_reader_processes"] != 2
                or normal["end_to_end_repeat_performed"] is not False
                or any(normal.get(k) != v for k, v in FLAGS.items())
                or [row_key(row) for row in normal["rows"]] != self.inventory(config)
                or [r["repeat"] for r in normal["repeats"]] != [0, 1]):
            raise ValueError("All normal rows and both fresh processes required before truth")
        repeated, contexts = [], {}
        for entry in normal["repeats"]:
            if entry["path"] != f"repeat-{entry['repeat']}.json" or digest(self.run / entry["path"]) != entry["sha256"]:
                raise ValueError("Repeat receipt changed")
            rep = read(self.run / entry["path"])
            if (rep["run_id"] != self.run_id or rep["repeat"] != entry["repeat"] or rep["state"] != "complete"
                    or rep["gt_read"] is not False or rep["pre_sha256"] != digest(self.run / "pre.json")
                    or any(rep.get(k) != v for k, v in FLAGS.items())
                    or [row_key(r) for r in rep["rows"]] != self.inventory(config, entry["repeat"])
                    or [a["case_id"] for a in rep["associations"]] != config["case_ids"]):
                raise ValueError("Full repeat inventory/provenance differs")
            repeated.extend(rep["rows"])
            for assoc in rep["associations"]:
                expected = f"records-r{entry['repeat']}/{assoc['case_id']}-association.json"
                if (assoc["path"] != expected or digest(self.run / expected) != assoc["sha256"]
                        or self.old.canonical_hash(read(self.run / expected)) != assoc["chain_context_sha256"]):
                    raise ValueError("Context receipt changed")
                contexts[entry["repeat"], assoc["case_id"]] = assoc["chain_context_sha256"]
        if repeated != normal["rows"]:
            raise ValueError("Repeat rows differ from combined normal")
        old, graphs, repeats, pairs = self.old_graphs(), [], {}, {}
        for row in normal["rows"]:
            key = row_key(row)
            expected = f"records-r{key[0]}/{key[1]}-{key[2]}-{key[3]}-{key[4]}.json"
            if row["path"] != expected or digest(self.run / expected) != row["sha256"]:
                raise ValueError("Normal graph receipt changed")
            graph = read(self.run / expected)
            graph_flags = {**FLAGS, "reader_algorithm_unchanged": key[-1] == "stroke_chain_axis"}
            if (row_key(graph) != key or graph["gt_read"] is not False or any(graph.get(k) != v for k, v in graph_flags.items())
                    or graph["chain_context_sha256"] != contexts[key[0], key[1]]
                    or graph["effective_policy"] != {**self.defaults()[key[-1]], **graph["call_policy"]}):
                raise ValueError("Normal graph identity/provenance changed")
            self.old.validate_anchor_input(graph)
            if key[-1] in CONTROLS:
                if physical_projection(graph) != physical_projection(old[key][1]):
                    raise ValueError("Original complete control graph changed")
            else:
                self.validate_new_pair(pairs[key[:-1]], graph)
            if key[-1] == "stroke_chain_axis":
                pairs[key[:-1]] = graph
            projection = physical_projection(graph, cross_repeat=True)
            if key[0] == 0:
                repeats[key[1:]] = projection
            elif repeats[key[1:]] != projection:
                raise ValueError("Full physical/audit graph differs across fresh repeats")
            graphs.append(graph)
        return normal, graphs

    def barrier(self):
        sys.addaudithook(self.old.block_truth)
        if (self.run / "normal-barrier.json").exists():
            raise FileExistsError("Preserve earlier barrier")
        started = time.perf_counter()
        before, config, index = self.checked(ancestors=True)
        normal, graphs = self.normal(config)
        result = dict(run_id=self.run_id, state="passed", gt_read=False,
            pre_sha256=digest(self.run / "pre.json"), inference_sha256=digest(self.run / "inference.json"),
            full_ancestor_receipts=len(index["ancestor_hashes"]), direct_receipts=len(index["hashes"]),
            own_receipts=len(before["hashes"]), normal_rows=len(graphs), repeat_pairs=len(graphs)//2,
            controls_equal=sum(g["reader"] in CONTROLS for g in graphs),
            elapsed_seconds=time.perf_counter()-started, **FLAGS)
        write(self.run / "normal-barrier.json", result)
        print("FOREGROUND_FULL_SHA_BARRIER", self.kind, len(normal["rows"]), flush=True)

    def require_authorization(self):
        authorization = read(self.run / "evaluation-authorization.json")
        expected = dict(run_id=self.run_id, state="passed", gt_read=False,
            pre_sha256=digest(self.run / "pre.json"), inference_sha256=digest(self.run / "inference.json"),
            barrier_sha256=digest(self.run / "normal-barrier.json"))
        if any(authorization.get(k) != v for k, v in expected.items()):
            raise ValueError("Independent cross-suite authorization required before truth")
        peer = Runner("g1_object" if self.kind == "fixture" else "fixture")
        bindings = []
        for runner in (self, peer):
            barrier = read(runner.run / "normal-barrier.json")
            binding = dict(run_id=runner.run_id, pre_sha256=digest(runner.run / "pre.json"),
                inference_sha256=digest(runner.run / "inference.json"))
            if (barrier["state"] != "passed" or barrier["gt_read"] is not False
                    or any(barrier.get(k) != v for k, v in binding.items())):
                raise ValueError("Both suites must complete normal barriers before truth")
            bindings.append(binding)
        if authorization["peer_barrier_sha256"] != digest(peer.run / "normal-barrier.json"):
            raise ValueError("Other suite barrier changed")
        audit = direct.checked_path(ROOT, authorization["independent_audit_path"])
        if digest(audit) != authorization["independent_audit_sha256"]:
            raise ValueError("Independent full normal audit changed")
        record = read(audit)
        if (record.get("state") != "passed" or record.get("gt_read") is not False
                or sorted(record.get("runs", []), key=lambda r: r["run_id"]) != sorted(bindings, key=lambda r: r["run_id"])):
            raise ValueError("Independent audit must bind both exact normal runs without GT")

    def evaluation_payload(self):
        self.require_authorization()
        before, config, _ = self.checked(ancestors=True)
        normal, graphs = self.normal(config)
        scene_id = read(self.point.CONFIG)["scene_run_id"]
        truth_path = ROOT / "data/eval_gt" / scene_id / "manifest.json"
        if digest(truth_path) != read(ROOT / ".runtime/experiments" / scene_id / "prepared.json")["truth_sha256"]:
            raise ValueError("Original full truth identity changed")
        truth_manifest = read(truth_path)
        manifest = read(self.point.INPUTS / "manifest.json")
        if (truth_manifest["input_sha256"] != digest(ROOT / "data/inputs" / scene_id / "manifest.json")
                or [case["case_id"] for case in truth_manifest["cases"]] != config["case_ids"]):
            raise ValueError("Full original truth/input/case inventory differs")
        for source, actual in zip(manifest["cases"], truth_manifest["cases"]):
            if [f["rgb_sha256"] for f in source["frames"]] != [f["rgb_sha256"] for f in actual["frames"]]:
                raise ValueError("Truth and source RGB frame order/identity differs")
        truths = {case["case_id"]: case for case in truth_manifest["cases"]}
        rows, native_rows, seen_native = [], [], {}
        for row, graph in zip(normal["rows"], graphs):
            declared = truths[row["case_id"]]["declared"]
            target = declared["target"]
            truth = target.get("segments", [target.get("endpoints")]) if target["present"] else []
            gaps = [declared["gap_segment"]] if "gap_segment" in declared else []
            score, error = objects.score_geometry(graph["state"], graph["segments"], truth, gaps)
            rows.append(dict(**row, graph=graph, physical=score, scoring_error=error,
                physical_equal_to_repeat_zero=None if row["repeat"] == 0 else True))
            key = row["case_id"], row["variant"]
            native_identity = {name: graph[name] for name in graph if name.startswith("native_")}
            if key in seen_native and seen_native[key] != native_identity:
                raise ValueError("Native arms cannot be deduplicated across differing repeats/readers/scales")
            if key not in seen_native:
                seen_native[key] = native_identity
                native, native_error = objects.score_geometry(graph["native_state"], graph["native_segments"], truth, gaps)
                native_rows.append(dict(case_id=key[0], variant=key[1], segments=graph["native_segments"],
                    physical=native, scoring_error=native_error, measurement_scope=graph["native_metric_scope"],
                    scored_without_readout_clipping=True))
        return dict(run_id=self.run_id, state="evaluated", pre_sha256=digest(self.run / "pre.json"),
            inference_sha256=digest(self.run / "inference.json"), truth_sha256=digest(truth_path),
            normal_barrier_sha256=digest(self.run / "normal-barrier.json"),
            evaluation_authorization_sha256=digest(self.run / "evaluation-authorization.json"),
            unchanged_own_receipts=len(before["hashes"]), rows=rows, native_rows=native_rows,
            physical_repeat_pairs=len(rows)//2, physical_repeat_equal=len(rows)//2,
            physical_policy=read(self.run / "evaluation-freeze.json")["physical_policy"],
            normal_elapsed_seconds=normal["elapsed_seconds"], end_to_end_repeat_performed=False,
            reader_qualified=False, g1_passed=False, **FLAGS)

    def evaluate(self):
        if self.public.exists() or (self.run / "evaluation.json").exists():
            raise FileExistsError("Preserve previous evaluation")
        result = self.evaluation_payload()
        write(self.run / "evaluation.json", result)
        write(self.public, result)
        print("FOREGROUND_EVALUATED", self.kind, len(result["rows"]), flush=True)

    def post(self):
        if self.audit.exists() or (self.run / "post.json").exists():
            raise FileExistsError("Preserve previous post audit")
        result = read(self.public)
        if result != self.evaluation_payload() or self.public.read_bytes() != (self.run / "evaluation.json").read_bytes():
            raise ValueError("Exact evaluation replay differs")
        audit = dict(run_id=self.run_id, state="passed", public_sha256=digest(self.public),
            exact_evaluation_replay=True, physical_repeat_pairs=result["physical_repeat_pairs"],
            native_geometry_unchanged=True, full_measurement_domain_preserved=True, g1_passed=False, **FLAGS)
        write(self.run / "post.json", audit)
        write(self.audit, audit)
        print("FOREGROUND_POST_PASSED", self.kind, flush=True)


def main(kind):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "infer", "infer-repeat", "barrier", "evaluate", "post"))
    parser.add_argument("repeat", type=int, nargs="?")
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    runner = Runner(kind)
    if args.stage == "infer-repeat":
        runner.infer_repeat(args.repeat)
    elif args.repeat is not None:
        parser.error("repeat is valid only for infer-repeat")
    else:
        getattr(runner, args.stage)()
