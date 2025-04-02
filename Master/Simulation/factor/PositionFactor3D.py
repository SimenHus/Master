from .common import *

class PositionFactor3D(CustomFactor):
    def __init__(self, key: int, measurement: Pose3, noise_model: noiseModel = None):
        super().__init__(noise_model, [key], self.evaluateError)
        self.measurement = measurement.translation()

    def evaluateError(self, _, values, H = None):
        T = values.atPose3(self.keys()[0])
        R = T.rotation().matrix()
        if H is not None:
            H[0] = np.array([
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ])
        return T.translation() - self.measurement
