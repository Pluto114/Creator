"""Coordinate-only Sim3 transforms for explicitly privileged evaluation diagnostics."""
from __future__ import annotations

import numpy as np


def similarity_parts(transform):
    value=np.asarray(transform,float)
    if value.shape!=(4,4) or not np.isfinite(value).all() or not np.allclose(value[3],[0,0,0,1],atol=1e-10,rtol=0):
        raise ValueError("Finite homogeneous similarity required")
    scale=float(np.cbrt(np.linalg.det(value[:3,:3])))
    if scale<=0:
        raise ValueError("Positive similarity scale required")
    rotation=value[:3,:3]/scale
    if not np.allclose(rotation@rotation.T,np.eye(3),atol=1e-8,rtol=0) or not np.isclose(np.linalg.det(rotation),1.,atol=1e-8,rtol=0):
        raise ValueError("Proper similarity rotation required")
    return scale,rotation,value[:3,3]


def transform_segments(segments,transform):
    scale,rotation,translation=similarity_parts(transform)
    points=np.asarray(segments,float)
    if points.size==0:
        return []
    if points.ndim!=3 or points.shape[1:]!=(2,3) or not np.isfinite(points).all():
        raise ValueError("Finite Nx2x3 segments required")
    return (scale*(points@rotation.T)+translation).tolist()


def transform_camera(extrinsic,transform):
    scale,rotation,translation=similarity_parts(transform)
    camera=np.asarray(extrinsic,float)
    if camera.shape==(4,4):
        if not np.allclose(camera[3],[0,0,0,1],atol=1e-10,rtol=0):
            raise ValueError("Invalid homogeneous camera")
        camera=camera[:3]
    if camera.shape!=(3,4) or not np.isfinite(camera).all():
        raise ValueError("Finite world-to-camera 3x4 or 4x4 required")
    q,u=camera[:,:3],camera[:,3]
    if not np.allclose(q@q.T,np.eye(3),atol=1e-8,rtol=0) or not np.isclose(np.linalg.det(q),1.,atol=1e-8,rtol=0):
        raise ValueError("Proper camera rotation required")
    moved=q@rotation.T
    # x_gt=sR x_old+t. Scale camera coordinates uniformly, preserving pixels.
    # Multiplying E by inverse(T) alone leaves a scaled 'rotation' in the API.
    return np.c_[moved,scale*u-moved@translation].tolist()
