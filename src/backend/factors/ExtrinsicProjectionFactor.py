
from gtsam import CustomFactor
from gtsam import noiseModel
from gtsam import Pose3

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.structs import Camera

class ExtrinsicProjectionFactor(CustomFactor):
    def __init__(self, ref_key: int, extrinsic_key: int, landmark_key: int, camera: 'Camera', measurement: np.ndarray, noise_model: noiseModel = None) -> None:
        super().__init__(noise_model, [ref_key, extrinsic_key, landmark_key], self.evaluateError)
        self.measurement = measurement
        self.camera = camera


    def evaluateError(self, _, values, H = None):
        Twr: Pose3 = values.atPose3(self.keys()[0])
        Trc: Pose3 = values.atPose3(self.keys()[1])
        landmark = values.atPoint3(self.keys()[2])

        H1 = np.zeros((6, 6), dtype=np.float64, order='F')
        H2 = np.zeros((6, 6), dtype=np.float64, order='F')
        H3 = np.zeros((3, 6), dtype=np.float64, order='F')
        H4 = np.zeros((3, 3), dtype=np.float64, order='F')

        Twc = Twr.compose(Trc, H1, H2)
        landmark_c = Twc.transformTo(landmark, H3, H4)
        error = self.camera.project(landmark_c) - self.measurement

        fx, fy = self.camera.parameters[:2]
        X, Y, Z = landmark_c
        projection_jac = 1 / Z * np.array([
                [fx, 0, -fx * X / Z],
                [0, fy, -fy * Y / Z]
        ], order='F')

        # Correct analytical solution
        # if H is not None:
        #     H[0] = projection_jac @ np.block([skew(landmark_c), -np.eye(3)]) @ Trc.inverse().AdjointMap()
        #     H[1] = projection_jac @ np.block([skew(landmark_c), -np.eye(3)])
        #     H[2] = projection_jac @ Twc.rotation().matrix().T
        
        if H is not None:
            H[0] = projection_jac @ H3 @ H1
            H[1] = projection_jac @ H3 @ H2
            H[2] = projection_jac @ H4

        return error
    