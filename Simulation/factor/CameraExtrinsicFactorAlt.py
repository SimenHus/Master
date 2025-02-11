
from .common import *

class CameraExtrinsicFactorAlt(CustomFactor):
    def __init__(self, camera_key: int, ref_key: int, landmark_key: int, measurement: np.ndarray, noise_model: noiseModel = None) -> None:
        super().__init__(noise_model, [camera_key, ref_key, landmark_key], self.evaluateError)
        self.measurement = measurement

    def evaluateError(self, _, values, H = None):
        T_world_ref = values.atPose3(self.keys()[0])
        T_ref_camera = values.atPose3(self.keys()[1])
        landmark = values.atPoint3(self.keys()[2])

        T_world_camera = T_ref_camera * T_world_ref
        landmark_camera = T_world_camera.transformFrom(landmark)

        predicted_bearing = landmark_camera
        predicted_range = np.linalg.norm(predicted_bearing).reshape((1,))
        predicted_bearing = predicted_bearing / predicted_range

        prediction = np.concatenate((predicted_range, predicted_bearing))
        error = prediction - self.measurement

        if H is not None:
            H[0] = np.zeros((4, 6))
            H[0][:3, :3] = np.eye(3)
            H[1] = np.zeros((4, 6))
            H[1][:3, :3] = np.eye(3)
            H[2] = np.vstack((
                landmark_camera / predicted_range,
                (np.eye(3) - landmark_camera@landmark_camera.T/predicted_range**2) / predicted_range
            ))

        return error