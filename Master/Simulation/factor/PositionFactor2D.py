from .common import *


class PositionFactor2D(CustomFactor):
    def __init__(self, key: int, measurement: Pose2, noise_model: noiseModel = None):
        super().__init__(noise_model, [key], self.evaluateError)
        self.measurement = measurement.translation()

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


