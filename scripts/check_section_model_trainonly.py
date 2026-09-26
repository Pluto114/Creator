"""Independent membership, training-frame and saved-score audit; never refits models."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments/src"))
from creator_eval.section_model_diagnostic import distances, summarize  # noqa: E402

RUN_ID="section-model-trainonly-v1-20260926r1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def identity(values):
    value=np.ascontiguousarray(values)
    return dict(shape=list(value.shape),dtype=value.dtype.str,sha256=hashlib.sha256(value.tobytes()).hexdigest())


def same_score(a,b):
    assert set(a)==set(b)
    for key in a:
        if isinstance(a[key],(float,int)):
            np.testing.assert_allclose(a[key],b[key],rtol=1e-12,atol=1e-12)
        else:
            assert a[key]==b[key],key


def audit():
    run=ROOT/".runtime/experiments"/RUN_ID
    inputs=ROOT/"data/inputs"/RUN_ID
    frozen=read(run/"prepared.json")
    parent=ROOT/".runtime/experiments"/frozen["parent_run_id"]
    assert digest(parent/"prepared.json")==frozen["parent_prepared_sha256"]
    cmd=["git","-c","safe.directory="+ROOT.as_posix(),"hash-object"]
    for name,sha in frozen["source_sha256"].items():
        assert digest(ROOT/name)==digest(run/"source_snapshot"/name)==sha,name
        raw=(ROOT/name).read_bytes()
        assert b"\r" not in raw and raw.endswith(b"\n") and not raw.endswith(b"\n\n")
        assert subprocess.check_output(cmd+["--no-filters",name],cwd=ROOT)==subprocess.check_output(cmd+["--path="+name,name],cwd=ROOT)
    assert digest(run/"source_freeze.json")==frozen["source_freeze_sha256"]
    assert read(run/"source_freeze.json")["source_sha256"]==frozen["source_sha256"]
    assert digest(run/"protocol.json")==frozen["protocol_sha256"]
    policy=read(run/"protocol.json")["method"]
    assert policy["split"]==dict(native_axis=2,grid_origin=[0.,0.,0.],block_cells=10,blocks=6,guard_cells_each_edge=2,first_cell=0)
    assert digest(inputs/"manifest.json")==frozen["input_sha256"]
    assert digest(ROOT/"data/eval_gt"/RUN_ID/"manifest.json")==frozen["truth_sha256"]
    truth=read(ROOT/"data/eval_gt"/RUN_ID/"manifest.json")
    old_truth_path=ROOT/"data/eval_gt"/frozen["parent_run_id"]/"manifest.json"
    assert digest(old_truth_path)==frozen["parent_truth_sha256"]
    assert truth["cases"]==read(old_truth_path)["cases"]
    manifest=read(inputs/"manifest.json")
    old_inputs=ROOT/"data/inputs"/frozen["parent_run_id"]
    assert digest(old_inputs/"manifest.json")==frozen["parent_input_sha256"]
    assert manifest["cases"]==read(old_inputs/"manifest.json")["cases"]
    baseline=frozen["previous_shared_result"]
    assert digest(ROOT/baseline["path"])==baseline["sha256"]
    old=read(ROOT/baseline["path"])
    assert old["source_sha256"]==frozen["previous_source_sha256"]
    old_rows={(r["case_id"],r["representation"],r["train_slices"][0],r["model"]):r for r in old["rows"]}
    index=read(run/"inference.json")
    assert index["state"]=="complete" and index["gt_read_during_inference"] is False and index["truth_read_tripwire_enabled"]
    assert index["source_sha256"]==frozen["source_sha256"] and index["input_sha256"]==frozen["input_sha256"]
    public=ROOT/"docs/experiments/results/2026-09-26-section-model-trainonly-r1.json"
    assert public.read_bytes()==(ROOT/"data/evaluation"/RUN_ID/"summary.json").read_bytes()
    summary=read(public)
    assert summary["inference_sha256"]==digest(run/"inference.json")
    assert summary["qualification"] is None and summary["emits_axis"] is False
    rows={(r["case_id"],r["representation"],r["fold"],r["model"]):r for r in summary["rows"]}
    cases={c["case_id"]:c for c in manifest["cases"]}
    assert len(cases)==len(manifest["cases"])==len(index["records"])==64
    assert {r["case_id"] for r in index["records"]}==set(cases)
    fold_records=[]
    checked_rows=0
    for entry in index["records"]:
        case=cases[entry["case_id"]]
        path=inputs/case["points_file"]
        assert digest(path)==digest(old_inputs/case["points_file"])==case["points_sha256"]==entry["input_sha256"]
        assert digest(run/entry["path"])==entry["sha256"]
        output=read(run/entry["path"])
        assert output["policy"]==policy and output["emits_axis"] is False
        points=np.unique(np.load(path,allow_pickle=False),axis=0)
        voxel=case["voxel_size"]
        cells=np.floor(points/voxel).astype(np.int64)
        # Independent literal interpretation of the frozen native block protocol.
        z=cells[:,2]
        native_block=np.floor_divide(z,10)
        keep=(z>=0)&(z<60)&(z%10>=2)&(z%10<=7)
        assigned=np.where((z<0)|(z>=60),-2,np.where(keep,native_block,-1))
        for fold in output["folds"]:
            parity=fold["fold"]
            train=np.isin(assigned,list(range(parity,6,2)))
            test=np.isin(assigned,list(range(1-parity,6,2)))
            train_cells=np.unique(cells[train],axis=0)
            test_cells=np.unique(cells[test],axis=0)
            assert set(map(tuple,train_cells)).isdisjoint(map(tuple,test_cells))
            assert identity(train_cells)==fold["train_cell_identity"] and identity(test_cells)==fold["heldout_cell_identity"]
            assert identity(points[train])==fold["train_point_identity"] and identity(points[test])==fold["heldout_point_identity"]
            for role,mask in dict(train=train,heldout=test,guard=assigned==-1,outside_window=assigned==-2).items():
                assert fold["role_points"][role]==int(mask.sum()) and fold["role_cells"][role]==len(np.unique(cells[mask],axis=0))
            z_gap=float(np.min(np.abs(np.unique(points[train,2])[:,None]-np.unique(points[test,2])))) if train.any() and test.any() else None
            assert z_gap is None or z_gap>=4*voxel-1e-12
            centers=(train_cells+.5)*voxel
            frame=fold["frame"]
            if len(centers)>=policy["model"]["minimum_fit_points"]:
                origin=centers.mean(axis=0)
                eigen,vectors=np.linalg.eigh((centers-origin).T@(centers-origin)/len(centers))
                for axis in range(3):
                    if vectors[np.argmax(np.abs(vectors[:,axis])),axis]<0:
                        vectors[:,axis]*=-1
                edges=np.quantile((centers-origin)@vectors[:,-1],[.05,.95])
                assert frame["state"]=="available" and frame["source"]=="training_occupied_voxel_centers_only"
                for actual,expected in ((frame["origin"],origin),(frame["vectors"],vectors),(frame["axial_quantiles_m"],edges)):
                    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-13)
                np.testing.assert_allclose(frame["maximum_radius_voxels"],(edges[1]-edges[0])/voxel*.25,rtol=0,atol=1e-13)
            else:
                assert frame["state"]=="unavailable"
            fold_records.append(dict(case_id=case["case_id"],fold=parity,shared_voxels=0,actual_minimum_z_gap_m=z_gap,
                role_points=fold["role_points"],role_cells=fold["role_cells"],frame_state=frame["state"]))
            for representation in output["representations"]:
                name=representation["name"]
                for item in [r for r in representation["rows"] if r["fold"]==parity]:
                    key=(case["case_id"],name,parity,item["model"])
                    row=rows[key]
                    assert row["accepted_model"] is False and row["training_preprocessing"]==fold
                    for field,value in item.items():
                        assert row[field]==value
                    tc,ti,tn=np.unique(cells[train],axis=0,return_inverse=True,return_counts=True)
                    training=(tc+.5)*voxel if name=="voxel_centers" else points[train]
                    train_weights=np.ones(len(tc)) if name=="voxel_centers" else 1./tn[ti]
                    assert identity(training)==item["train_representation_identity"]
                    assert identity(train_weights)==item["training_weights_identity"]
                    assert float(train_weights.sum())==item["training_occupied_weight"]
                    if item["fit"]["state"]=="fitted":
                        train_values=(training-frame["origin"])@np.array(frame["vectors"])/voxel
                        train_score=summarize(distances(item["model"],train_values[:,:2],item["fit"],policy["model"]),train_weights)
                        same_score(train_score,item["fit"]["training_residual"])
                    previous=old_rows[key]
                    assert row["previous_shared"]["heldout_rmse_voxels"]==previous["heldout_rmse_voxels"]
                    old_rmse=previous["heldout_rmse_voxels"]
                    scores=[]
                    for slot in item["test_slices"]:
                        chosen=points[assigned==slot]
                        cc,ii,counts=np.unique(np.floor(chosen/voxel).astype(np.int64),axis=0,return_inverse=True,return_counts=True)
                        cloud=(cc+.5)*voxel if name=="voxel_centers" else chosen
                        weight=np.ones(len(cc)) if name=="voxel_centers" else 1./counts[ii]
                        if not len(cloud):
                            score=dict(state="missing_test_points")
                        elif item["fit"]["state"]!="fitted":
                            score=dict(state="not_scored_missing_fit",point_count=len(cloud))
                        else:
                            values=(cloud-frame["origin"])@np.array(frame["vectors"])/voxel
                            score=summarize(distances(item["model"],values[:,:2],item["fit"],policy["model"]),weight)
                        expected=dict(slice=slot,**score)
                        same_score(expected,item["heldout"][len(scores)])
                        scores.append(score)
                    complete=all("rmse_voxels" in x for x in scores)
                    rmse=float(np.sqrt(sum(s["rmse_voxels"]**2*s["occupied_weight"] for s in scores)/sum(s["occupied_weight"] for s in scores))) if complete else None
                    if rmse is None:
                        assert row["heldout_rmse_voxels"] is None and row["rmse_change_from_shared_voxels"] is None
                    else:
                        np.testing.assert_allclose(row["heldout_rmse_voxels"],rmse,rtol=1e-12,atol=1e-12)
                        np.testing.assert_allclose(row["rmse_change_from_shared_voxels"],rmse-old_rmse,rtol=1e-12,atol=1e-12)
                    checked_rows+=1
    assert checked_rows==len(rows)==len(summary["rows"])==768
    assert len(summary["comparisons"])==256 and all(r["selected_model"] is None for r in summary["comparisons"])
    original=ROOT/".runtime/experiments"/frozen["engineering_revision"]["previous_run_id"]
    original_frozen=read(original/"prepared.json")
    original_index=read(original/"inference.json")
    assert read(original/"protocol.json")["method"]==policy
    revised=[]
    for name,sha in original_frozen["source_sha256"].items():
        assert digest(original/"source_snapshot"/name)==sha
        if digest(ROOT/name)!=sha:
            revised.append(name)
    assert set(revised)=={"scripts/run_section_model_trainonly.py","scripts/check_section_model_trainonly.py",
        "configs/section_model_trainonly_v1.json","tests/test_section_model_trainonly.py"}
    original_records={r["case_id"]:r for r in original_index["records"]}
    for entry in index["records"]:
        original_entry=original_records[entry["case_id"]]
        assert digest(original/original_entry["path"])==original_entry["sha256"]==entry["sha256"]
    original_summary=read(ROOT/"data/evaluation"/frozen["engineering_revision"]["previous_run_id"]/"summary.json")
    assert original_summary["rows"]==summary["rows"] and original_summary["comparisons"]==summary["comparisons"]
    result=dict(state="passed",run_id=RUN_ID,frozen_sources=len(frozen["source_sha256"]),
        identical_parent_inputs=64,identical_parent_labels=64,verified_inference_records=64,
        preserved_v1_source_snapshots=len(original_frozen["source_sha256"]),v1_revised_live_sources=sorted(revised),
        v1_identical_inference_record_bytes=64,v1_identical_model_rows=768,
        v1_prepared_sha256=digest(original/"prepared.json"),
        independently_verified_native_folds=len(fold_records),shared_training_heldout_voxels=0,
        recomputed_saved_model_score_rows=checked_rows,paired_previous_rows=checked_rows,
        public_runtime_bytes_identical=True,gt_read_during_inference=False,fold_records=fold_records,
        summary_sha256=digest(public),audit_source_sha256=digest(__file__),source_sha256=frozen["source_sha256"],
        scope="Checks fixed native grouping, training-only frames, saved predictions and provenance; no optimizer refit, classification or blind validation claim")
    destination=ROOT/"docs/experiments/results/2026-09-26-section-model-trainonly-r1-audit.json"
    with destination.open("x",encoding="utf-8",newline="\n") as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write("\n")
    print("AUDITED_SECTION_TRAINONLY",len(fold_records),"folds;",checked_rows,"saved score rows; zero shared voxels",flush=True)


if __name__=="__main__":
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    audit()
