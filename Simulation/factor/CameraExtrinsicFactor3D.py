
from .common import *

class CameraExtrinsicFactor3D(CustomFactor):
    def __init__(self, from_key: int, to_key: int, measurement: Pose3, noise_model: noiseModel = None) -> None:
        super().__init__(noise_model, [from_key, to_key], self.evaluateError)
        self.measurement = measurement

    def evaluateError(self, _, values, H = None):
        T_rel = values.atPose3(self.keys()[0]) # Current estimate of relative pose between base and camera
        T_cam = values.atPose3(self.keys()[1]) # Current estimate of camera pose in world frame

        H_rel = np.zeros((6, 6), dtype=np.float64, order='F')
        H_cam = np.zeros((6, 6), dtype=np.float64, order='F')

        prediction = T_cam.compose(T_rel.inverse(), H_cam, H_rel) # Camera pose - relative pose = base pose
        error = Pose3.localCoordinates(prediction, self.measurement) # Compare measured base pose with predicted base pose

        if H is not None:
            H[0] = H_cam # From gtsam math.pdf, compose derivative first argument
            H[1] = np.eye(6) # From gtsam math.pdf, compose derivative second argument

        return error