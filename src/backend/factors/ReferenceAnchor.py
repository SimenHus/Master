
from gtsam import CustomFactor
from gtsam import Pose3
from gtsam.utils.numerical_derivative import numericalDerivative31, numericalDerivative32

import numpy as np


def f(Trc, Twc, meas):
    return Pose3.localCoordinates(meas, Twc.compose(Trc.inverse()))

class ReferenceAnchor(CustomFactor):
    def __init__(self, rel_key: int, cam_key: int, measurement: Pose3, noise_model=None):
        super().__init__(noise_model, [rel_key, cam_key], self.evaluateError)
        self.measurement = measurement

    def evaluateError(self, _, values, H=None):
        Trc = values.atPose3(self.keys()[0])
        Twc = values.atPose3(self.keys()[1])

        H1 = np.zeros((6, 6), dtype=np.float64, order='F')
        H2 = np.zeros((6, 6), dtype=np.float64, order='F')
        H3 = np.zeros((6, 6), dtype=np.float64, order='F')
        HLocal = np.zeros((6, 6), dtype=np.float64, order='F')


        # Forward solution
        # prediction = self.measurement.compose(Trc, H1, H2)
        # error = Pose3.localCoordinates(Twc, prediction, HLocal)

        # if H is not None:
        #     H[0] = -HLocal@H2
        #     H[1] = HLocal

        # Inverse solution
        prediction = Twc.compose(Trc.inverse(H1), H2, H3)
        error = Pose3.localCoordinates(self.measurement, prediction, HLocal)

        error = f(Trc, Twc, self.measurement)

        if H is not None:
            H[0] = numericalDerivative31(f, Trc, Twc, self.measurement)
            H[1] = numericalDerivative32(f, Trc, Twc, self.measurement)

        # if H is not None:
        #     H[0] = -HLocal@H3@H1
        #     H[1] = -HLocal@H2

        return error
