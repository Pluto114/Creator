"""Direct CAD-world physical evaluation; no registration or rod refitting."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .line_controls import _segments, curve_metrics, gap_coverage, point_to_segments_distance

POLICY=dict(curve_tolerance_m=.025,curve_spacing_m=.002,gap_endpoint_guard_m=.025,
    endpoint_numeric_relative_tolerance=1e-10,coordinate_policy="declared_fixture_world_m_no_alignment")


def endpoint_inventory(segments):
    values=_segments(segments)
    points=values.reshape(-1,3)
    if not len(points):
        return dict(native_endpoint_count=0,boundary_endpoint_count=0,boundary_points=[],shared_degree_two_points=[],junctions=[],
            scope="Endpoint-incidence boundary; does not split intersections inside segments")
    epsilon=POLICY["endpoint_numeric_relative_tolerance"]*max(1.,float(np.abs(points).max()))
    groups=[]
    for index,point in enumerate(points):
        selected=next((g for g in groups if np.linalg.norm(point-points[g[0]])<=epsilon),None)
        if selected is None:
            groups.append([index])
        else:
            selected.append(index)
    boundaries=[points[g[0]].tolist() for g in groups if len(g)==1]
    shared=[points[g[0]].tolist() for g in groups if len(g)==2]
    junctions=[dict(point=points[g[0]].tolist(),degree=len(g)) for g in groups if len(g)>2]
    return dict(native_endpoint_count=len(points),boundary_endpoint_count=len(boundaries),boundary_points=boundaries,
        shared_degree_two_points=shared,junctions=junctions,numeric_merge_tolerance_m=epsilon,
        scope="Endpoint-incidence boundary; degree-two joins are not physical endpoints; interior intersections are not split")


def distance_summary(values,reason=None):
    values=np.asarray(values,float)
    valid=len(values)>0 and np.isfinite(values).all()
    return dict(count=len(values),distances_m=values.tolist() if valid else [None]*len(values),
        mean_m=float(values.mean()) if valid else None,max_m=float(values.max()) if valid else None,
        p95_m=float(np.quantile(values,.95)) if valid else None,reason=None if valid else reason or "empty_or_unavailable")


def endpoint_set_report(source,target):
    source=np.asarray(source,float).reshape(-1,3)
    target=np.asarray(target,float).reshape(-1,3)
    distances=np.linalg.norm(source[:,None]-target[None],axis=2).min(axis=1) if len(source) and len(target) else np.full(len(source),np.inf)
    return distance_summary(distances,"missing_source_or_target_boundary")


def score_finite_structure(predicted,truth,gaps=()):
    prediction=_segments(predicted,"prediction")
    ground_truth=_segments(truth,"truth")
    curves=curve_metrics(prediction,ground_truth,tolerance=POLICY["curve_tolerance_m"],spacing=POLICY["curve_spacing_m"])
    # Old curve validation rejects positive-length collinear overlap first.
    # Thus the remaining finite-segment lengths add to union arc length; touching
    # endpoints do not add length, and splitting a good rod cannot earn extra weight.
    pred_length=float(np.linalg.norm(prediction[:,1]-prediction[:,0],axis=1).sum())
    truth_length=float(np.linalg.norm(ground_truth[:,1]-ground_truth[:,0],axis=1).sum())
    p,t=endpoint_inventory(prediction),endpoint_inventory(ground_truth)
    pp=np.asarray(p["boundary_points"]).reshape(-1,3)
    tp=np.asarray(t["boundary_points"]).reshape(-1,3)
    paired=dict(state="boundary_count_mismatch",pairs=[],maximum_error_m=None)
    if len(pp)==len(tp) and len(pp):
        cost=np.linalg.norm(pp[:,None]-tp[None],axis=2)
        left,right=linear_sum_assignment(cost)
        errors=cost[left,right]
        paired=dict(state="scored",pairs=[dict(prediction_boundary_index=int(i),truth_boundary_index=int(j),
            distance_m=float(cost[i,j])) for i,j in zip(left,right)],maximum_error_m=float(errors.max()),
            matching="Evaluation-only endpoint-distance bijection; no geometry movement")
    elif not len(pp) and not len(tp):
        paired["state"]="no_boundary_endpoints"
    gap_rows=[gap_coverage(prediction,gap,tolerance=POLICY["curve_tolerance_m"],spacing=POLICY["curve_spacing_m"],
        endpoint_guard=POLICY["gap_endpoint_guard_m"]) for gap in gaps]
    interior_length=sum(row["guarded_interior"]["source_length"] for row in gap_rows)
    occupied_length=sum(row["guarded_interior"]["covered_length"] for row in gap_rows)
    raw_truth_endpoints=ground_truth.reshape(-1,3)
    return dict(state="scored",policy=POLICY,alignment_performed=False,geometry_modified=False,
        empty_prediction=not len(prediction),prediction_segment_count=len(prediction),truth_segment_count=len(ground_truth),
        curves=curves,length=dict(prediction_union_m=pred_length,truth_union_m=truth_length,
            signed_error_m=pred_length-truth_length,absolute_error_m=abs(pred_length-truth_length),
            relative_error=None if truth_length==0 else (pred_length-truth_length)/truth_length,
            policy="Positive-length collinear overlap rejected; native finite-segment arc sum, no endpoint-distance surrogate"),
        predicted_endpoints=p,truth_endpoints=t,
        truth_boundary_to_prediction_boundary=endpoint_set_report(tp,pp),
        prediction_boundary_to_truth_boundary=endpoint_set_report(pp,tp),
        raw_truth_endpoints_to_prediction_curve=distance_summary(point_to_segments_distance(raw_truth_endpoints,prediction),"empty_prediction_or_truth"),
        boundary_bijection=paired,gaps=gap_rows,
        guarded_gap=dict(interior_length_m=interior_length,false_proximity_length_m=occupied_length,
            false_proximity_fraction=occupied_length/interior_length if interior_length else None,
            scope="Length within tolerance of predicted finite curves inside guarded GT gap; not proof of topological connectivity"),
        scope="Declared CAD-world finite-structure diagnostics. Empty output has zero positive-truth recovery and undefined precision; no qualification gate")
