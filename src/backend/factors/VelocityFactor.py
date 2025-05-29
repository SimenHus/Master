
import gtsam
import numpy as np
from gtsam import Pose3, Rot3, Point3, CustomFactor
from gtsam.utils.numerical_derivative import numericalDerivative31, numericalDerivative32

def f(T1, T2, twist):
    return Pose3.localCoordinates(T2, twist.compose(T1))

class VelocityFactor(CustomFactor):
    def __init__(self, pose_from, pose_to, measurement, noise_model=None):
        super().__init__(noise_model, [pose_from, pose_to], self.evaluateError)
        self.measurement = measurement

    def evaluateError(self, _, values, H=None):
        Twx1: Pose3 = values.atPose3(self.keys()[0])
        Twx2: Pose3 = values.atPose3(self.keys()[1])

        H1 = np.zeros((6, 6), dtype=np.float64, order='F')
        H2 = np.zeros((6, 6), dtype=np.float64, order='F')
        H3 = np.zeros((6, 6), dtype=np.float64, order='F')
        HLocal = np.zeros((6, 6), dtype=np.float64, order='F')

        xi_w = self.measurement
        twist = Pose3.Expmap(xi_w, H1)
        prediction = twist.compose(Twx1, H2, H3)
        error = Pose3.localCoordinates(Twx2, prediction, HLocal)

        error = f(Twx1, Twx2, twist)

        if H is not None:
            H[0] = numericalDerivative31(f, Twx1, Twx2, twist)
            H[1] = numericalDerivative32(f, Twx1, Twx2, twist)

        # if H is not None:
        #     H[0] = -HLocal@H3
        #     H[1] = HLocal

        return error