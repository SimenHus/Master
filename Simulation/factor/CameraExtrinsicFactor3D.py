
from .common import *

class CameraExtrinsicFactor3D(CustomFactor):
    def __init__(self, from_key: int, to_key: int, measurement: Pose3, noise_model: noiseModel = None) -> None:
        super().__init__(noise_model, [from_key, to_key], self.evaluateError)
        self.measurement = measurement

    def evaluateError(self, _, values, H = None):
        T_ref_camera = values.atPose3(self.keys()[0]) # Current estimate of camera extrinsics from reference frame
        T_world_camera = values.atPose3(self.keys()[1]) # Current estimate of camera in world frame

        T1 = T_world_camera
        T2 = T_ref_camera

        prediction = T1.inverse() * T2
        error = Pose3.localCoordinates(self.measurement, prediction) # Error

        if H is not None:
            H[0] = Pose3.AdjointMap(prediction.inverse()).T # From gtsam math.pdf
            H[1] = -np.eye(6) # From gtsam math.pdf

        return error