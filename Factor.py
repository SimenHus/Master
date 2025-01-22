
from common import *

# https://gtsam.org/tutorials/intro.html



class GNSSFactor(gtsam.CustomFactor):
    def __init__(self, key: int, measurement: np.ndarray, noise_model: gtsam.noiseModel = None):
        super().__init__(noise_model, [key], self.evaluateError)
        self.measurement = measurement

    def evaluateError(self, _, values, H = None):
        T = values.atPose2(self.keys()[0])
        R = T.rotation()
        if H is not None:
            # H[0] = np.array([
            #     [R.c(), -R.s(), 0.0],
            #     [R.s(), R.c(), 0.0]
            # ])
            H[0] = np.array([
                [1, 0, 0],
                [0, 1, 0]
            ])
        return T.translation() - self.measurement
    

class GNSSFactor3D(gtsam.CustomFactor):
    def __init__(self, key: int, measurement: np.ndarray, noise_model: gtsam.noiseModel = None):
        super().__init__(noise_model, [key], self.evaluateError)
        self.measurement = measurement

    def evaluateError(self, _, values, H = None):
        T = values.atPose3(self.keys()[0])
        R = T.rotation().matrix()
        if H is not None:
            # H[0] = np.array([
            #     [R.c(), -R.s(), 0.0],
            #     [R.s(), R.c(), 0.0]
            # ])
            H[0] = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0]
            ])
            # H[0] = np.hstack([
            #     R, np.zeros((3, 3))
            # ])
        return T.translation() - self.measurement