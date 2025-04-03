
from gtsam import CustomFactor
from gtsam import noiseModel
from gtsam import Pose3, Point3

import numpy as np
import symforce.symbolic as sf


# This line is a nonsense line

class SymbolicJacobian:
    T_rel = sf.Pose3(
        R = sf.Rot3.symbolic('R_rel'),
        t = sf.V3.symbolic('t_rel')
    )
    T_ref = sf.Pose3(
        R = sf.Rot3.symbolic('R_ref'),
        t = sf.V3.symbolic('t_ref')
    )
    landmark = sf.V3.symbolic('L')
    T_cam = T_ref.compose(T_rel)
    pred = T_cam.inverse() * landmark

    @classmethod
    def wrt_camera(clc, T_rel: Pose3, T_ref: Pose3, landmark: Point3) -> np.ndarray:
        jac = clc.pred.jacobian(clc.T_rel)

        return clc.subs(T_rel, T_ref, landmark, jac)
    
    @classmethod
    def wrt_reference(clc, T_rel: Pose3, T_ref: Pose3, landmark: Point3) -> np.ndarray:
        jac = clc.pred.jacobian(clc.T_ref)

        return clc.subs(T_rel, T_ref, landmark, jac)
    
    @classmethod
    def wrt_landmark(clc, T_rel: Pose3, T_ref: Pose3, landmark: Point3) -> np.ndarray:
        jac = clc.pred.jacobian(clc.landmark)

        return clc.subs(T_rel, T_ref, landmark, jac)

    @classmethod
    def subs(clc, T_rel: Pose3, T_ref: Pose3, landmark: Point3, jac):
        subs = {}

        # Substitute rotation components
        subs.update({k: v for k, v in zip(clc.T_rel.R.to_storage(), T_rel.rotation().toQuaternion().coeffs())})
        subs.update({k: v for k, v in zip(clc.T_ref.R.to_storage(), T_ref.rotation().toQuaternion().coeffs())})

        # Substitute translation components
        subs.update({k: v for k, v in zip(clc.T_rel.t, T_rel.translation())})
        subs.update({k: v for k, v in zip(clc.T_ref.t, T_ref.translation())})

        # Substitute landmark components
        subs.update({k: v for k, v in zip(clc.landmark, landmark)})

        return jac.subs(subs).to_numpy()


class CameraExtrinsicFactor(CustomFactor):
    def __init__(self, camera_key: int, ref_key: int, landmark_key: int, measurement: np.ndarray, noise_model: noiseModel = None) -> None:
        super().__init__(noise_model, [camera_key, ref_key, landmark_key], self.evaluateError)
        self.measurement = measurement

    def evaluateError(self, _, values, H = None):
        T_rel = values.atPose3(self.keys()[0])
        T_ref = values.atPose3(self.keys()[1])
        landmark = values.atPoint3(self.keys()[2])


        H_rel = np.zeros((6, 6), dtype=np.float64, order='F')
        H_ref = np.zeros((6, 6), dtype=np.float64, order='F')
        H_cam = np.zeros((3, 6), dtype=np.float64, order='F')
        H_p = np.zeros((3, 3), dtype=np.float64, order='F')


        T_cam = T_ref.compose(T_rel, H_ref, H_rel)
        prediction = T_cam.transformTo(landmark, H_cam, H_p) # Landmark in camera frame

        error = prediction - self.measurement

        if H is not None:
            H[0] = SymbolicJacobian.wrt_camera(T_rel, T_ref, landmark)
            H[1] = SymbolicJacobian.wrt_reference(T_rel, T_ref, landmark)
            H[2] = SymbolicJacobian.wrt_landmark(T_rel, T_ref, landmark)

        # if H is not None:
        #     H[0] = H_cam@H_rel # Problem in the second part of this result?
        #     H[1] = H_cam@H_ref # Seems correct
        #     H[2] = H_p # Correct, same as T_cam.inverse().rotation().matrix()

        return error
    