import gtsam
import numpy as np
from gtsam import Pose3, NonlinearFactorGraph, Values
from gtsam.symbol_shorthand import X

# Create the factor graph
graph = NonlinearFactorGraph()

# Define noise model
noise_model = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)  # SE(3) 6D error term

# Define the unknown extrinsic transform (T_ref^cam) as a variable
initial_estimate = Values()
extrinsic_guess = Pose3()  # Identity as initial guess
initial_estimate.insert(X(0), extrinsic_guess)

# Example data: known reference motions and camera motions
reference_motions = [Pose3.Expmap(np.random.randn(6) * 0.1) for _ in range(5)]
camera_motions = [Pose3.Expmap(np.random.randn(6) * 0.1) for _ in range(5)]

# Add factors based on motion constraints
for i in range(len(reference_motions)):
    ref_motion = reference_motions[i]
    cam_motion = camera_motions[i]

    # Construct the error term for T_ref^cam consistency
    factor = gtsam.BetweenFactorPose3(
        X(0), X(0), ref_motion.inverse() * cam_motion, noise_model
    )
    
    graph.add(factor)

# Optimize using Levenberg-Marquardt
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
result = optimizer.optimize()

# Extract the estimated extrinsic transformation
estimated_extrinsic = result.atPose3(X(0))
print("Estimated Extrinsic Transform (T_ref^cam):")
print(estimated_extrinsic)
