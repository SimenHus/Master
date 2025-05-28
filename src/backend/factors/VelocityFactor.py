
import gtsam
import numpy as np
from gtsam import Pose3, Rot3, Point3, CustomFactor

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

        xi_x1 = self.measurement

        prediction = Twx1.compose(Pose3.Expmap(xi_x1, H1), H2, H3)
        error = Pose3.localCoordinates(Twx2, prediction, HLocal)

        if H is not None:
            H[0] = -HLocal@H2
            H[1] = HLocal

        return error