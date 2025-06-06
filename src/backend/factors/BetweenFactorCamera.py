
from gtsam import CustomFactor
from gtsam import Pose3
from gtsam.utils.numerical_derivative import numericalDerivative31, numericalDerivative32, numericalDerivative33
import numpy as np


def f(Twr, Trc, Twc):
    return Pose3.localCoordinates(Twc, Twr.compose(Trc))

class BetweenFactorCamera(CustomFactor):
    def __init__(self, ref_key: int, rel_key: int, cam_key: int, noise_model=None):
        super().__init__(noise_model, [ref_key, rel_key, cam_key], self.evaluateError)

    def evaluateError(self, _, values, H=None):
        Twr: Pose3 = values.atPose3(self.keys()[0])
        Trc: Pose3 = values.atPose3(self.keys()[1])
        Twc: Pose3 = values.atPose3(self.keys()[2])
        
        error = f(Twr, Trc, Twc)

        if H is not None:
            H[0] = numericalDerivative31(f, Twr, Trc, Twc)
            H[1] = numericalDerivative32(f, Twr, Trc, Twc)
            H[2] = numericalDerivative33(f, Twr, Trc, Twc)

        return error
