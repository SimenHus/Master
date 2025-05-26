
from gtsam import CustomFactor
from gtsam import Pose3

import numpy as np


class ReferenceAnchor(CustomFactor):
    def __init__(self, rel_key: int, cam_key: int, measurement: Pose3, noise_model=None):
        super().__init__(noise_model, [rel_key, cam_key], self.evaluateError)
        self.measurement = measurement

    def evaluateError(self, _, values, H=None):
        T_rel = values.atPose3(self.keys()[0])
        T_cam = values.atPose3(self.keys()[1])

        H_meas = np.zeros((6, 6), dtype=np.float64, order='F')
        H_rel = np.zeros((6, 6), dtype=np.float64, order='F')
        HLocal = np.zeros((6, 6), dtype=np.float64, order='F')


        # Forward solution
        prediction = self.measurement.compose(T_rel, H_meas, H_rel)
        error = Pose3.localCoordinates(T_cam, prediction, HLocal)

        if H is not None:
            H[0] = -HLocal@H_rel
            H[1] = HLocal

        return error
