
from gtsam import CustomFactor
from gtsam import noiseModel
from gtsam import Pose3

from gtsam.utils.numerical_derivative import numericalDerivative51, numericalDerivative52, numericalDerivative53

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.structs import Camera


def f(Twr, Trc, landmark, camera, measurement):
    Twc = Twr.compose(Trc)
    landmark_c = Twc.transformTo(landmark)
    return camera.project(landmark_c) - measurement

class ExtrinsicProjectionFactorNumeric(CustomFactor):
    def __init__(self, ref_key: int, extrinsic_key: int, landmark_key: int, camera: 'Camera', measurement: np.ndarray, noise_model: noiseModel = None) -> None:
        super().__init__(noise_model, [ref_key, extrinsic_key, landmark_key], self.evaluateError)
        self.measurement = measurement
        self.camera = camera


    def evaluateError(self, _, values, H = None):
        Twr: Pose3 = values.atPose3(self.keys()[0])
        Trc: Pose3 = values.atPose3(self.keys()[1])
        landmark = values.atPoint3(self.keys()[2])

        error = f(Twr, Trc, landmark, self.camera, self.measurement)
        
        if H is not None:
            H[0] = numericalDerivative51(f, Twr, Trc, landmark, self.camera, self.measurement)
            H[1] = numericalDerivative52(f, Twr, Trc, landmark, self.camera, self.measurement)
            H[2] = numericalDerivative53(f, Twr, Trc, landmark, self.camera, self.measurement)

        return error
    