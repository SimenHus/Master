import cv2
import numpy as np
import yaml
from src.frontend.feature_extraction import extract_features
from src.frontend.visual_odometry import estimate_motion
from src.backend.factor_graph import SLAMOptimizer
from src.mapping.point_cloud import PointCloudMap

# Load configuration
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Camera Intrinsics
K = np.array([[config["camera"]["fx"], 0, config["camera"]["cx"]],
              [0, config["camera"]["fy"], config["camera"]["cy"]],
              [0, 0, 1]])

# SLAM Components
slam_optimizer = SLAMOptimizer()
map_builder = PointCloudMap()

# Load images
image1 = cv2.imread("datasets/sample_images/1.png", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("datasets/sample_images/2.png", cv2.IMREAD_GRAYSCALE)

# Extract features
kp1, desc1 = extract_features(image1)
kp2, desc2 = extract_features(image2)

# Estimate motion
R, t = estimate_motion(kp1, kp2, K)

# Add to factor graph
slam_optimizer.add_pose_factor(0, gtsam.Pose3())
slam_optimizer.add_pose_factor(1, gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(t[0], t[1], t[2])))

# Optimize
result = slam_optimizer.optimize()
print("Optimized Pose:", result.atPose3(1))

# Generate point cloud (dummy data)
points = np.random.rand(100, 3)
map_builder.add_points(points)
map_builder.visualize()
