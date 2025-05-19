import numpy as np
import gtsam
from gtsam import CustomFactor, NonlinearFactorGraph, noiseModel, Values, symbol, NonlinearEqualityPose3
from gtsam import Pose3, Point3, Rot3, LevenbergMarquardtParams, LevenbergMarquardtOptimizer, Cal3_S2, SmartProjectionParams
from gtsam import SmartProjectionPoseFactorCal3_S2
from gtsam.utils import plot
import matplotlib.pyplot as plt

# --- BetweenFactorMod as before ---
class BetweenFactorMod(CustomFactor):
    def __init__(self, ref_key: int, rel_key: int, cam_key: int, noise_model=None):
        super().__init__(noise_model, [ref_key, rel_key, cam_key], self.evaluateError)

    def evaluateError(self, _, values, H=None):
        T_ref = values.atPose3(self.keys()[0])
        T_rel = values.atPose3(self.keys()[1])
        T_cam = values.atPose3(self.keys()[2])

        H_ref = np.zeros((6, 6), dtype=np.float64, order='F')
        H_rel = np.zeros((6, 6), dtype=np.float64, order='F')
        H_cam = np.zeros((6, 6), dtype=np.float64, order='F')

        prediction = Pose3.between(T_ref, T_cam, H_ref, H_cam)
        error = Pose3.localCoordinates(T_rel, prediction, H_rel)

        if H is not None:
            H[0] = H_rel @ H_ref
            H[1] = H_rel
            H[2] = H_rel @ H_cam

        return error

# --- Graph Setup ---
graph = NonlinearFactorGraph()
initial = Values()

# Calibration
K = Cal3_S2(500, 500, 0, 320, 240)
pose_noise = noiseModel.Diagonal.Sigmas(np.ones(6) * 0.1)

# Smart factor parameters
smart_params = SmartProjectionParams()
# smart_params.setLinearizationMode(SmartProjectionParams.CHOLESKY)
# smart_params.setDegeneracyMode(SmartProjectionParams.ZERO_ON_DEGENERACY)

# Use this noise model for pixel observations
pixel_noise = noiseModel.Isotropic.Sigma(2, 1.0)

# Unknown camera extrinsics to estimate
rel_key = symbol('e', 0)
T_rel_true = Pose3.Expmap([0.0, 0.0, 0.0, 0.5, 0.0, 0.2])
initial.insert(rel_key, Pose3.Expmap([0.0, 0.0, 0.0, 0.3, 0.1, 0.2]))
# initial.insert(rel_key, T_rel_true)

# Assume we're observing one landmark from 3 camera frames
N = 3
smart_factor = SmartProjectionPoseFactorCal3_S2(pixel_noise, K, smart_params)

for i in range(N):
    # Known reference pose
    ref_key = symbol('r', i)
    T_ref = Pose3(Rot3.RzRyRx(0.0, 0.0, 0.1*i), Point3(1.0*i, 0, 0))
    initial.insert(ref_key, T_ref)
    graph.add(NonlinearEqualityPose3(ref_key, T_ref))

    # Camera pose (to be estimated)
    cam_key = symbol('c', i)
    T_cam_guess = T_ref.compose(initial.atPose3(rel_key))
    initial.insert(cam_key, T_cam_guess)

    # Tie T_cam_i = T_ref_i * T_rel via your custom factor
    graph.add(BetweenFactorMod(ref_key, rel_key, cam_key, pose_noise))

    # Simulate 3D point projected into camera i
    landmark_world = Point3(1.5, 0.0, 1.0)
    T_cam = T_ref.compose(T_rel_true)
    camera = gtsam.PinholeCameraCal3_S2(T_cam, K)
    uv = camera.project(landmark_world)

    # Add this observation to the smart factor
    smart_factor.add(uv, cam_key)

# Add the smart factor (shared for all 3 observations)
graph.add(smart_factor)

# Optimize
params = LevenbergMarquardtParams()
# params.setVerbosity("ERROR")
optimizer = LevenbergMarquardtOptimizer(graph, initial, params)
result = optimizer.optimize()

# Output estimated transform
print("\nEstimated camera extrinsic (T_rel):")
print(result.atPose3(rel_key))


from src.visualization import GraphVisualization

GraphVisualization.FactorGraphVisualization.draw_factor_graph('', graph, initial)