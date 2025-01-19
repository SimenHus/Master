import gtsam


import numpy as np
from gtsam import NonlinearFactorGraph, noiseModel, Values
from gtsam import (Point3, Pose3)

# 🔧 1. Define the custom factor for extrinsic calibration
class CameraExtrinsicFactor(gtsam.CustomFactor):
    def __init__(self, key_base_pose, key_extrinsic, observed_point, measured_point, noise_model):
        super().__init__(noise_model, key_base_pose, key_extrinsic)
        self.observed_point = observed_point  # Point in the camera frame
        self.measured_point = measured_point  # Expected point in the base frame

    def evaluateError(self, base_pose, extrinsic_transform, H1=None, H2=None):
        """
        Calculate the error between the observed point (transformed from camera to base frame)
        and the expected point in the base frame.
        """
        # Transform the observed point from the camera frame to the base frame
        transformed_point = base_pose.compose(extrinsic_transform).transformFrom(self.observed_point)

        # Calculate the error as the difference between the transformed point and the measured point
        error = transformed_point - self.measured_point

        # If Jacobians are requested, set them to identity for simplicity
        if H1 is not None:
            H1[0] = np.eye(3)
        if H2 is not None:
            H2[0] = np.eye(3)

        return error

# 🔧 2. Create a graph and add the custom factor
graph = NonlinearFactorGraph()

# Define prior for the base pose (robot's pose in the world frame)
prior_noise = noiseModel.Diagonal.Sigmas((0.1, 0.1, 0.1, 0.1, 0.1, 0.1))
graph.add(gtsam.PriorFactorPose3(0, Pose3(), prior_noise))

# Define an initial guess for the extrinsic parameters (translation + rotation)
initial_extrinsic = Pose3(gtsam.Rot3.RzRyRx(0.1, 0.2, -0.1), Point3(0.5, 0.1, 0.2))


# Define the observed and expected points
observed_point = Point3(1.0, 0.5, 0.3)  # Point in the camera frame
measured_point = Point3(1.2, 0.6, 0.4)  # Expected point in the base frame

# Create the custom extrinsic factor
extrinsic_noise = noiseModel.Diagonal.Sigmas(np.array([0.05, 0.05, 0.05]))
graph.add(CameraExtrinsicFactor(0, 1, observed_point, measured_point, extrinsic_noise))

# 🔧 3. Create initial estimates
initial_estimate = Values()
initial_estimate.insert(0, Pose3())  # Initial guess for the base pose
initial_estimate.insert(1, initial_extrinsic)  # Initial guess for the extrinsic parameters

# 🔧 4. Optimize the graph
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
result = optimizer.optimize()

# 🔧 5. Print the optimized results
print("Optimized Base Pose:", result.atPose3(0))
print("Optimized Extrinsic Parameters:", result.atPose3(1))
