
from gtsam import CustomFactor
from gtsam import Pose3
from gtsam.utils.numerical_derivative import numericalDerivative41, numericalDerivative42, numericalDerivative43
import numpy as np


def f(Twc1, Twc2, Trc, A):
    B = Pose3.between(Twc1, Twc2)
    X = Trc
    return Pose3.localCoordinates(A.compose(X), X.compose(B))

class HandEyeFactor(CustomFactor):
    def __init__(self, from_key, to_key, extrinsic_key, state_from, state_to, noise_model=None):
        super().__init__(noise_model, [from_key, to_key, extrinsic_key], self.evaluateError)
        self.state_from = state_from
        self.state_to = state_to

    def evaluateError(self, _, values, H=None):
        Twc1: Pose3 = values.atPose3(self.keys()[0])
        Twc2: Pose3 = values.atPose3(self.keys()[1])
        Trc: Pose3 = values.atPose3(self.keys()[2])
        Twr1: Pose3 = self.state_from.pose
        Twr2: Pose3 = self.state_to.pose

        H1 = np.zeros((6, 6), dtype=np.float64, order='F')
        H2 = np.zeros((6, 6), dtype=np.float64, order='F')
        H3 = np.zeros((6, 6), dtype=np.float64, order='F')
        H4 = np.zeros((6, 6), dtype=np.float64, order='F')
        H5 = np.zeros((6, 6), dtype=np.float64, order='F')
        H6 = np.zeros((6, 6), dtype=np.float64, order='F')
        H7 = np.zeros((6, 6), dtype=np.float64, order='F')
        H8 = np.zeros((6, 6), dtype=np.float64, order='F')
        HLocal = np.zeros((6, 6), dtype=np.float64, order='F')


        A = Pose3.between(Twr1, Twr2)
        B = Pose3.between(Twc1, Twc2, H1, H2)
        X = Trc

        # error = Pose3.localCoordinates(X.compose(B, H5, H6), A.compose(X, H3, H4), HLocal)
        error = f(Twc1, Twc2, Trc, A)
        if H is not None:
            H[0] = numericalDerivative41(f, Twc1, Twc2, Trc, A)
            H[1] = numericalDerivative42(f, Twc1, Twc2, Trc, A)
            H[2] = numericalDerivative43(f, Twc1, Twc2, Trc, A)

        # if H is not None:
        #     H[0] = HLocal@H6@H1
        #     H[1] = HLocal@H6@H2
        #     H[2] = HLocal@(H5-H4)

        return error
