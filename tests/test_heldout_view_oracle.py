"""Geometry invariance fixtures; no generated or private experiment artifacts."""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments/src"))
from creator_eval.camera_bundle import project  # noqa: E402
from creator_eval.heldout_view_oracle import (  # noqa: E402
    similarity_parts,
    transform_camera,
    transform_segments,
)


class OracleCoordinateTests(unittest.TestCase):
    def fixture(self):
        transform=np.eye(4)
        transform[:3,:3]=2.3*Rotation.from_euler("xyz",[.2,-.3,.1]).as_matrix()
        transform[:3,3]=[1.,-2.,.7]
        camera=np.c_[Rotation.from_euler("xyz",[.1,.05,-.2]).as_matrix(),[.3,-.1,6.]]
        segments=np.array([[[-.2,.1,0.],[.2,.1,1.]],[[.2,.1,1.3],[.3,.1,2.]]])
        k=np.array([[450.,0.,319.5],[0.,460.,239.5],[0.,0.,1.]])
        return transform,camera,segments,k

    def test_joint_transform_preserves_pixels_and_scales_depth(self):
        transform,camera,segments,k=self.fixture()
        before,depth=project(segments.reshape(-1,3),k,camera)
        after,new_depth=project(np.array(transform_segments(segments,transform)).reshape(-1,3),k,np.array(transform_camera(camera,transform)))
        np.testing.assert_allclose(before,after,atol=1e-10,rtol=0)
        np.testing.assert_allclose(new_depth,depth*2.3,atol=1e-10,rtol=0)

    def test_camera_center_transforms_with_segments(self):
        transform,camera,_,_=self.fixture()
        updated=np.array(transform_camera(camera,transform))
        center=-camera[:,:3].T@camera[:,3]
        expected=transform[:3,:3]@center+transform[:3,3]
        np.testing.assert_allclose(-updated[:,:3].T@updated[:,3],expected,atol=1e-12,rtol=0)
        np.testing.assert_allclose(updated[:,:3]@updated[:,:3].T,np.eye(3),atol=1e-12,rtol=0)

    def test_identity_and_homogeneous_camera_agree(self):
        _,camera,segments,_=self.fixture()
        homogeneous=np.eye(4)
        homogeneous[:3]=camera
        np.testing.assert_allclose(transform_camera(homogeneous,np.eye(4)),camera,atol=0,rtol=0)
        np.testing.assert_allclose(transform_segments(segments,np.eye(4)),segments,atol=0,rtol=0)

    def test_inverse_round_trip_keeps_gap_and_endpoints(self):
        transform,camera,segments,_=self.fixture()
        inverse=np.linalg.inv(transform)
        np.testing.assert_allclose(transform_segments(transform_segments(segments,transform),inverse),segments,atol=1e-12,rtol=0)
        np.testing.assert_allclose(transform_camera(transform_camera(camera,transform),inverse),camera,atol=1e-12,rtol=0)
        self.assertEqual(len(transform_segments(segments,transform)),2)

    def test_empty_segments_remain_empty(self):
        self.assertEqual(transform_segments([],np.eye(4)),[])

    def test_reflection_shear_and_zero_scale_are_rejected(self):
        for linear in (np.diag([-1.,1.,1.]),np.diag([2.,1.,1.]),np.zeros((3,3))):
            transform=np.eye(4)
            transform[:3,:3]=linear
            with self.assertRaises(ValueError):
                similarity_parts(transform)

    def test_nonfinite_or_malformed_geometry_is_rejected(self):
        for segments in ([[1.,2.,3.]],np.full((1,2,3),np.nan)):
            with self.assertRaises(ValueError):
                transform_segments(segments,np.eye(4))
        with self.assertRaises(ValueError):
            transform_camera(np.zeros((3,4)),np.eye(4))


if __name__=="__main__":
    unittest.main()
