
from .common import *

def skew_symmetric(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def Hp(p):
    return np.block([
        [skew_symmetric(-p), np.eye(3)],
        [np.zeros((1,3)), np.zeros((1,3))]
    ])

def Hinv(T, p):
    Tinvp = T.transformTo(p)
    return np.block([
        [skew_symmetric(-Tinvp), np.eye(3)],
        [np.zeros((1,3)), np.zeros((1,3))]
    ])

class CameraExtrinsicFactorAlt(CustomFactor):
    def __init__(self, camera_key: int, ref_key: int, landmark_key: int, measurement: np.ndarray, noise_model: noiseModel = None) -> None:
        super().__init__(noise_model, [camera_key, ref_key, landmark_key], self.evaluateError)
        self.measurement = measurement

    def evaluateError(self, _, values, H = None):
        T_ref = values.atPose3(self.keys()[0])
        T_rel = values.atPose3(self.keys()[1])
        landmark = values.atPoint3(self.keys()[2])

        T_cam = T_ref.compose(T_rel)
        prediction = T_cam.transformTo(landmark) # Landmark in camera frame

        landmark_temp = T_ref.transformTo(landmark)
        error = prediction - self.measurement
        # error = self.measurement - prediction
        H1 = Hinv(T_ref, landmark)
        H2 = Hinv(T_rel, landmark_temp)
        if H is not None:
            # H[0] = -(T_rel.inverse().matrix()@Hp2)[:3, :]
            # H[1] = -(T_rel.inverse().matrix()@T_ref.inverse().matrix()@Hp)[:3, :]
            # H[0] = (-T_rel.inverse().matrix()@H1)[:3, :]
            # H[1] = (-H2)[:3, :]
            H[0] = np.zeros((3, 6))
            H[1] = np.zeros((3, 6))
            H[2] = T_cam.inverse().rotation().matrix()

        return error