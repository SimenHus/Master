
from gtsam import CustomFactor
from gtsam import Pose3

import numpy as np


class BetweenFactorCamera(CustomFactor):
    def __init__(self, ref_key: int, rel_key: int, cam_key: int, noise_model=None):
        super().__init__(noise_model, [ref_key, rel_key, cam_key], self.evaluateError)

    def evaluateError(self, _, values, H=None):
        Twr: Pose3 = values.atPose3(self.keys()[0])
        Trc: Pose3 = values.atPose3(self.keys()[1])
        Twc: Pose3 = values.atPose3(self.keys()[2])

        H1 = np.zeros((6, 6), dtype=np.float64, order='F')
        H2 = np.zeros((6, 6), dtype=np.float64, order='F')
        HLocal = np.zeros((6, 6), dtype=np.float64, order='F')

        # Compose solution
        prediction = Twr.compose(Trc, H1, H2)
        error = Pose3.localCoordinates(Twc, prediction, HLocal)

        if H is not None:
            H[0] = -HLocal@H1
            H[1] = -HLocal@H2
            H[2] = HLocal


        return error
