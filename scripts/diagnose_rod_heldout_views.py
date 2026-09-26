"""After-normal privileged K/pose attribution; never changes the normal decision."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import run_rod_heldout_views as normal
from prepare_rod_reference import source_snapshot
from run_camera_bundle_pilot import ROOT, digest, read_json, write_json
from run_rod_candidate_ablation import canonical_hash

# isort: split
from creator_eval.heldout_view_oracle import transform_camera, transform_segments
from creator_eval.rod_validation_evidence import check_target_anchors

RUN_ID='rod-heldout-view-oracle-v1-20260926'
RUN=ROOT/'.runtime/experiments'/RUN_ID
PUBLIC=ROOT/'docs/experiments/results/2026-09-26-heldout-view-oracle.json'
AUDIT=ROOT/'docs/experiments/results/2026-09-26-heldout-view-oracle-audit.json'
INDEPENDENT=ROOT/'.runtime/experiments/rod-heldout-view-audit-v1-20260926'
OWN_SOURCES={'scripts/diagnose_rod_heldout_views.py','experiments/src/creator_eval/heldout_view_oracle.py',
    'tests/test_heldout_view_oracle.py'}
ARMS=('estimated','true_K_only','true_pose_only','true_K_and_pose')
POLICY=dict(arms=list(ARMS),threshold_px=2.,alignment='Unchanged old full-control camera-only Sim3',
    no_refit=True,no_geometry_change=True,no_normal_output_write=True,no_old_rejection_promotion=True,
    proposal_timing='After observing normal retention outcomes; post-hoc privileged evaluation, never a normal inference method')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def block_truth(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes)):
        path=str(args[0]).replace('\\','/').lower()
        if '/data/eval_gt/' in path:
            raise PermissionError('New truth may only be opened after oracle source/normal receipts are frozen')


def prepare():
    import subprocess
    sys.addaudithook(block_truth)
    if RUN.exists() or PUBLIC.exists() or AUDIT.exists():
        raise FileExistsError('Preserve existing diagnostic attempts')
    before=read_json(INDEPENDENT/'before-evaluation.json')
    assert before['state']=='passed' and not before['gt_read'] and not before['physical_scores_read']
    assert normal.PUBLIC.exists() and (normal.RUN/'evaluation.json').exists()
    assert digest(normal.PUBLIC)==digest(normal.RUN/'evaluation.json')
    config,_,_,receipts=normal.audit_normal()
    assert read_json(normal.RUN/'before-evaluation.json')['receipts']==receipts
    assert digest(normal.PHYSICAL)==normal.PHYSICAL_SHA
    for path in (normal.PUBLIC,normal.RUN/'evaluation.json',normal.RUN/'before-evaluation.json',normal.PHYSICAL,
        INDEPENDENT/'prepared.json',INDEPENDENT/'before-evaluation.json',normal.SCENE/'prepared.json',normal.SCENE_INPUTS/'manifest.json'):
        receipts[path.relative_to(ROOT).as_posix()]=digest(path)
    command=['git','-c','safe.directory='+ROOT.as_posix(),'hash-object']
    for name in OWN_SOURCES:
        raw=(ROOT/name).read_bytes()
        assert b'\r' not in raw and raw.endswith(b'\n') and not raw.endswith(b'\n\n')
        assert subprocess.check_output(command+['--no-filters',name],cwd=ROOT)==subprocess.check_output(command+['--path='+name,name],cwd=ROOT)
    RUN.mkdir(parents=True)
    sources=source_snapshot(RUN,set(config['source_sha256'])|OWN_SOURCES)
    write_json(RUN/'source_freeze.json',dict(source_sha256=sources,policy=POLICY,created_at_utc=stamp(),before_new_truth_read=True))
    scene=read_json(normal.SCENE/'prepared.json')
    write_json(RUN/'prepared.json',dict(run_id=RUN_ID,normal_run_id=normal.RUN_ID,source_sha256=sources,
        source_freeze_sha256=digest(RUN/'source_freeze.json'),policy=POLICY,normal_receipts=receipts,
        normal_config_sha256=digest(normal.RUN/'method_config.json'),normal_inference_sha256=digest(normal.RUN/'inference.json'),
        normal_evaluation_sha256=digest(normal.PUBLIC),independent_before_sha256=digest(INDEPENDENT/'before-evaluation.json'),
        scene_prepared_sha256=digest(normal.SCENE/'prepared.json'),expected_truth_sha256=scene['truth_sha256'],
        new_truth_read_during_prepare=False,created_at_utc=stamp()))
    print('ORACLE_PREPARED',len(sources),'sources;',len(receipts),'normal receipts',flush=True)


def checked():
    prepared=read_json(RUN/'prepared.json')
    assert prepared['run_id']==RUN_ID and prepared['policy']==POLICY
    assert digest(RUN/'source_freeze.json')==prepared['source_freeze_sha256']
    assert read_json(RUN/'source_freeze.json')['source_sha256']==prepared['source_sha256']
    for name,sha in prepared['source_sha256'].items():
        assert digest(ROOT/name)==digest(RUN/'source_snapshot'/name)==sha,name
    for name,sha in prepared['normal_receipts'].items():
        assert digest(ROOT/name)==sha,name
    return prepared


def make_output():
    frozen=checked()
    _,manifest=normal.checked()
    _,_,records,_=normal.audit_normal()
    public=read_json(normal.PUBLIC)
    assert public['state']=='complete' and public['run_id']==normal.RUN_ID
    old_physical=read_json(normal.PHYSICAL)
    truth_path=ROOT/'data/eval_gt'/normal.SCENE_ID/'manifest.json'
    assert digest(truth_path)==frozen['expected_truth_sha256']
    truth=read_json(truth_path)
    assert truth['input_sha256']==digest(normal.SCENE_INPUTS/'manifest.json')
    gt_cases={c['case_id']:c for c in truth['cases']}
    old_rows={(r['task_id'],r['condition']):r for r in public['rows']}
    rows,anchors,transforms=[],[],{}
    for record in records:
        task=next(t for t in manifest['tasks'] if t['task_id']==record['task_id'])
        physical=next(f for f in old_physical['rows'] if (f['parent'],f['case_id'],f['method'])==(record['parent'],record['case_id'],'baseline'))
        transform=np.asarray(physical['physical']['alignment']['prediction_world_to_gt_world'])
        transforms[record['task_id']]=transform.tolist()
        gt=gt_cases[record['case_id']]
        true_cameras={c['view_id']:c for c in gt['cameras']}
        true_frames={f['view_id']:f for f in gt['frames']}
        frames_by_arm={arm:[] for arm in ARMS}
        accepted=[]
        for frame,pose in zip(task['new_frames'],record['poses']):
            vid=frame['view_id']
            true=true_cameras[vid]
            assert vid==pose['view_id'] and frame['rgb_sha256']==true_frames[vid]['rgb_sha256']==pose['source']['rgb_sha256']
            estimated_e=pose['proposal'].get('E')
            estimated_e=transform_camera(estimated_e,transform) if estimated_e is not None else None
            estimated_k=pose['proposal'].get('K')
            accepted.append(pose['verification']['state']=='validated' and not any(x['role']=='validation' for x in pose['missing']))
            for arm in ARMS:
                k=true['K_index'] if arm in ('true_K_only','true_K_and_pose') else estimated_k
                e=true['world_to_camera_cv'] if arm in ('true_pose_only','true_K_and_pose') else estimated_e
                frames_by_arm[arm].append(dict(view_id=vid,size_wh=frame['size_wh'],source_sha256=frame['rgb_sha256'],
                    source_kind='rgb',K_index=k,world_to_camera_cv=e))
        for rod in record['rods']:
            family=next(f for f in task['families'] if f['task_id']==rod['task_id'])
            packet=next(p for p in family['packets'] if p['condition_id']==record['condition'])
            segments=(packet['identity'] or {'segments':[]})['segments']
            assert canonical_hash(segments)==rod['geometry_sha256']
            aligned=transform_segments(segments,transform)
            old=old_rows[(rod['task_id'],record['condition'])]
            assert old['state']==rod['state']
            eligible=rod['old_identity_state']=='accepted' and rod['old_camera_state']=='candidate_camera_correction' and all(accepted)
            for arm in ARMS:
                check=check_target_anchors(frames_by_arm[arm],aligned,task['query']['anchors'],threshold_px=2.)
                state=normal.final_state(rod['old_identity_state'],rod['old_camera_state'],accepted,check['state'])
                if not eligible:
                    assert state!='retained_candidate'
                if arm=='estimated':
                    assert check['state']==rod['target_check']['state'] and state==rod['state']
                    for a,b in zip(check['per_anchor'],rod['target_check']['per_anchor']):
                        assert a['view_id']==b['view_id'] and a['state']==b['state']
                        for key in ('distance_px','minimum_distance_bound_px','maximum_distance_bound_px'):
                            if a[key] is None:
                                assert b[key] is None
                            else:
                                np.testing.assert_allclose(a[key],b[key],atol=1e-8,rtol=0)
                row=dict(task_id=rod['task_id'],camera_task_id=record['task_id'],parent=record['parent'],case_id=record['case_id'],
                    condition=record['condition'],method=rod['method'],arm=arm,normal_state=rod['state'],original_eligible=eligible,
                    old_identity_state=rod['old_identity_state'],old_camera_state=rod['old_camera_state'],
                    geometry_sha256=rod['geometry_sha256'],aligned_geometry_sha256=canonical_hash(aligned),
                    unchanged_control_sim3_sha256=canonical_hash(transform),target_check=check,
                    diagnostic_counterfactual_state=state,normal_decision_modified=False,
                    normal_physical=old['physical'],frames=frames_by_arm[arm])
                rows.append(row)
                for anchor in check['per_anchor']:
                    anchors.append(dict(task_id=rod['task_id'],condition=record['condition'],method=rod['method'],
                        case_id=record['case_id'],arm=arm,normal_state=rod['state'],original_eligible=eligible,**anchor))
    assert len(rows)==240 and len(anchors)==480 and len(transforms)==6
    summary={}
    for arm in ARMS:
        selected=[r for r in rows if r['arm']==arm]
        eligible=[r for r in selected if r['original_eligible']]
        excluded=[r for r in selected if not r['original_eligible']]
        summary[arm]=dict(rod_count=len(selected),anchor_count=sum(r['arm']==arm for r in anchors),
            eligible_count=len(eligible),old_excluded_count=len(excluded),
            all_target_states=dict(Counter(r['target_check']['state'] for r in selected)),
            eligible_target_states=dict(Counter(r['target_check']['state'] for r in eligible)),
            diagnostic_counterfactual_states=dict(Counter(r['diagnostic_counterfactual_state'] for r in selected)),
            old_exclusion_promotions=sum(r['diagnostic_counterfactual_state']=='retained_candidate' for r in excluded))
    checked()
    return dict(state='complete_privileged_diagnostic',run_id=RUN_ID,normal_run_id=normal.RUN_ID,policy=POLICY,
        source_sha256=frozen['source_sha256'],prepared_sha256=digest(RUN/'prepared.json'),
        normal_evaluation_sha256=frozen['normal_evaluation_sha256'],truth_sha256=digest(truth_path),
        before_truth_source_freeze_sha256=frozen['source_freeze_sha256'],rod_count=60,unique_anchor_claims=6,
        per_arm_anchor_uses=120,rod_arm_rows=240,anchor_arm_rows=480,old_control_transforms=transforms,
        summaries=summary,rows=rows,anchors=anchors,no_normal_output_modified=True,no_geometry_refit=True,
        scope='Post-hoc privileged attribution on six new views of three old Blender layouts. GT K/pose arms are unavailable to normal use; no new identity, camera, or 3D accuracy acceptance rule.')


def evaluate():
    if PUBLIC.exists() or (RUN/'evaluation.json').exists():
        raise FileExistsError('Preserve oracle evaluation')
    output=make_output()
    write_json(RUN/'evaluation.json',output)
    write_json(PUBLIC,output)
    print('ORACLE_EVALUATED',output['summaries'],flush=True)


def check():
    if AUDIT.exists():
        raise FileExistsError('Preserve oracle audit')
    stored=read_json(PUBLIC)
    assert digest(PUBLIC)==digest(RUN/'evaluation.json')
    recomputed=make_output()
    assert canonical_hash(stored)==canonical_hash(recomputed)
    receipt=dict(state='passed',run_id=RUN_ID,prepared_sha256=digest(RUN/'prepared.json'),
        evaluation_sha256=digest(PUBLIC),recomputed_rod_arm_rows=240,recomputed_anchor_arm_rows=480,
        normal_receipts_unchanged=True,estimated_arm_agrees_with_normal=True,old_exclusion_promotions=0,
        audit_source_sha256=digest(__file__),
        scope='Same frozen arithmetic replay and identity audit plus separate Sim3 projection invariance unit tests; not an independent implementation of the scorer')
    write_json(AUDIT,receipt)
    print('ORACLE_CHECKED 240 rod-arm / 480 anchor-arm rows; normal receipts unchanged',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','evaluate','check'))
    args=parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    globals()[args.stage]()
