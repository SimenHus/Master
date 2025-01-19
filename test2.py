import gtsam

import gtsam.noiseModel
import numpy as np


from Factor import DistanceFactor2D, CameraExtrinsicFactor



graph = gtsam.NonlinearFactorGraph()

key = gtsam.symbol('x', 0)

prior_noise = gtsam.noiseModel.Diagonal.Sigmas((0.1, 0.1, 0.1, 0.1, 0.1, 0.1))
prior_value = gtsam.Pose3() # Empty transformation matrix T
graph.add(gtsam.PriorFactorPose3(0, prior_value, prior_noise))


observed_point = gtsam.Point3(1.0, 0.5, 0.3)
expected_point = gtsam.Point3(1.2, 0.6, 0.4)

extrinsic_noise = gtsam.noiseModel.Diagonal.Sigmas((0.05, 0.05, 0.05))
graph.add(CameraExtrinsicFactor(0, 1, extrinsic_noise, observed_point, expected_point))


initial_extrinsic = gtsam.Pose3(gtsam.Rot3.RzRyRx(0.1, 0.2, -0.1), gtsam.Point3(0.5, 0.1, 0.2))
initial_estimate = gtsam.Values()
initial_estimate.insert(0, gtsam.Pose3())  # Initial guess for the base pose
initial_estimate.insert(1, initial_extrinsic)  # Initial guess for the extrinsic parameters

optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
result = optimizer.optimize()

print("Optimized Base Pose:", result.atPose3(0))
print("Optimized Extrinsic Parameters:", result.atPose3(1))