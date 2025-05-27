
import gtsam
import numpy as np
from gtsam import Pose3, Rot3, Point3, CustomFactor
from gtsam.utils.numerical_derivative import numericalDerivative21


def f(T12, xi):
    return T12.inverse().AdjointMap() @ xi

class KinematicCameraFactor(CustomFactor):
    def __init__(self, pose_from, pose_to, extrinsics_key, measurement, dt, noise_model=None):
        super().__init__(noise_model, [pose_from, pose_to, extrinsics_key], self.evaluateError)
        self.measurement = measurement
        self.dt = dt

    def evaluateError(self, _, values, H=None):
        Twc1: Pose3 = values.atPose3(self.keys()[0])
        Twc2: Pose3 = values.atPose3(self.keys()[1])
        Trc: Pose3 = values.atPose3(self.keys()[2])
        dt = self.dt

        H1 = np.zeros((6, 6), dtype=np.float64, order='F')
        H2 = np.zeros((6, 6), dtype=np.float64, order='F')
        H3 = np.zeros((6, 6), dtype=np.float64, order='F')
        H4 = np.zeros((6, 6), dtype=np.float64, order='F')
        HLocal = np.zeros((6, 6), dtype=np.float64, order='F')

        xi_r1 = self.measurement.twist * dt
        xi_c1 = f(Trc, xi_r1)

        J = numericalDerivative21(f, Trc, xi_r1)

        prediction = Pose3.between(Twc1, Twc2, H1, H2)
        error = Pose3.localCoordinates(Pose3.Expmap(xi_c1, H4), prediction, HLocal)

        if H is not None:
            H[0] = -HLocal@H1
            H[1] = -HLocal@H2
            # H[2] = HLocal@H4@H3
            H[2] = -H4@J

        return error
