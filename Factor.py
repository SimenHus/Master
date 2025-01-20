
from common import *

# https://gtsam.org/tutorials/intro.html



class GNSSFactor(gtsam.CustomFactor):
    def __init__(self, measurement: 'np.ndarray[2]', keys: list = [], noise_model: gtsam.noiseModel = None):
        super().__init__(noise_model, keys, self.evaluateError)
        self.measurement = measurement

    def evaluateError(self, _, values, H = None):
        x, y = self.measurement
        T = values.atPose2(self.keys()[0])
        R = T.rotation()
        if H is not None:
            H[0] = np.array([
                [R.c(), -R.s(), 0.0],
                [R.s(), R.c(), 0.0]
            ])
        return np.array([T.x() - x, T.y() - y])