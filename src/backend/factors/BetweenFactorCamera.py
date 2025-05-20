
from gtsam import CustomFactor
from gtsam import Pose3

import numpy as np


class BetweenFactorCamera(CustomFactor):
    def __init__(self, ref_key: int, rel_key: int, cam_key: int, noise_model=None):
        super().__init__(noise_model, [ref_key, rel_key, cam_key], self.evaluateError)

    def evaluateError(self, _, values, H=None):
        T_ref = values.atPose3(self.keys()[0])
        T_rel = values.atPose3(self.keys()[1])
        T_cam = values.atPose3(self.keys()[2])

        H_ref = np.zeros((6, 6), dtype=np.float64, order='F')
        H_rel = np.zeros((6, 6), dtype=np.float64, order='F')
        H_cam = np.zeros((6, 6), dtype=np.float64, order='F')

        prediction = Pose3.between(T_ref, T_cam, H_ref, H_cam)
        error = Pose3.localCoordinates(T_rel, prediction, H_rel)

        # TESTED NUMERICALLY
        if H is not None:
            H[0] = -H_rel @ H_ref
            H[1] = H_rel
            H[2] = -H_rel @ H_cam

        return error
