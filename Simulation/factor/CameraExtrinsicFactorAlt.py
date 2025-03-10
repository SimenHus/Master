
from .common import *

import gtsam

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
        T_rel = values.atPose3(self.keys()[0])
        T_ref = values.atPose3(self.keys()[1])
        landmark = values.atPoint3(self.keys()[2])


        H_rel = np.zeros((6, 6), dtype=np.float64, order='F')
        H_ref = np.zeros((6, 6), dtype=np.float64, order='F')
        H_cam = np.zeros((3, 6), dtype=np.float64, order='F')
        H_p = np.zeros((3, 3), dtype=np.float64, order='F')


        T_cam = T_ref.compose(T_rel, H_ref, H_rel)
        prediction = T_cam.transformTo(landmark, H_cam, H_p) # Landmark in camera frame

        error = prediction - self.measurement

        # print(error)
        if H is not None:
            H[0] = H_cam@H_rel # Problem in the second part of this result?
            H[1] = H_cam@H_ref # Seems correct
            H[2] = H_p # Correct, same as T_cam.inverse().rotation().matrix()

        return error