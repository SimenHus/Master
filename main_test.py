
import gtsam
from gtsam import ISAM2Params, ISAM2, Marginals
from gtsam import NonlinearFactorGraph, Values, Symbol
from gtsam import PriorFactorPose3, BetweenFactorPose3, BearingRangeFactor3D, PriorFactorPoint3, Pose3, Point3, Rot3, GPSFactor
from gtsam import noiseModel, LevenbergMarquardtOptimizer, DoglegOptimizer
from gtsam.symbol_shorthand import X, L, T

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from Visualization.GraphVisualization import FactorGraphVisualization
from Visualization.PlotVisualization import plot_graph3D

from Simulation.factor import CameraExtrinsicFactorAlt

from src.backend import SLAM


slam_fg = SLAM()


landmarks_world = [
    Point3(2, 0, 0),
    Point3(-2, 0, 0)
]

rotation_gt = [ # Construct ground truth rotations from quaternion
    Rot3(1, 0, 0, 0),
    Rot3(1, 0, 0, 0),
    Rot3(1, 0, 0, 0)
]

position_gt = [
    Point3(0, 0, 0),
    Point3(0, 1, 0),
    Point3(0, 2, 0)
]

position_noisy = [
    Point3(0.2, -0.1, 0.1),
    Point3(-0.1, 0.3, -0.3),
    Point3(0.2, -0.1, -0.1)
]

position_noise = noiseModel.Diagonal.Sigmas([0.1, 0.1, 0.1])
pose_noise = noiseModel.Diagonal.Sigmas([0.1, 0.1, 0.1, 0.05, 0.05, 0.05])

odometry = Pose3(Rot3(), Point3(0, 1, 0))
odometry_noise = noiseModel.Diagonal.Sigmas([0.2, 0.2, 0.2, 0.1, 0.1, 0.1]) # Odometry noise model

pos = position_gt[0]
rot = rotation_gt[0]

slam_fg.pose_measurement(1, Pose3(rot, pos), pose_noise)

for i in range(1, len(position_gt)):
    pos = position_gt[i] # Include noise in inital estimates
    rot = rotation_gt[i]
    slam_fg.pose_measurement(i+1, Pose3(rot, pos), pose_noise)
    slam_fg.odometry_measurement(i, i+1, odometry, odometry_noise)
    slam_fg.optimize()

camera_pos = Point3(1, 1, 1)
camera_rot = Rot3(1, 0, 0, 0)
camera_extrinsics = Pose3(camera_rot, camera_pos)

extrinsic_factor_noise = noiseModel.Diagonal.Sigmas([0.1, 0.1, 0.1])
slam_fg.landmark_measurement(1, 1, landmarks_world[0], (Pose3(rotation_gt[0], position_gt[0]).compose(camera_extrinsics)).transformTo(landmarks_world[0]), extrinsic_factor_noise)
slam_fg.optimize()
slam_fg.landmark_measurement(2, 1, landmarks_world[0], (Pose3(rotation_gt[1], position_gt[1]).compose(camera_extrinsics)).transformTo(landmarks_world[0]), extrinsic_factor_noise)
slam_fg.optimize()
slam_fg.landmark_measurement(2, 2, landmarks_world[1], (Pose3(rotation_gt[1], position_gt[1]).compose(camera_extrinsics)).transformTo(landmarks_world[1]), extrinsic_factor_noise)
slam_fg.optimize()
slam_fg.landmark_measurement(3, 2, landmarks_world[1], (Pose3(rotation_gt[2], position_gt[2]).compose(camera_extrinsics)).transformTo(landmarks_world[1]), extrinsic_factor_noise)
slam_fg.optimize()
        

# result = LevenbergMarquardtOptimizer(graph, initial).optimize()
result = slam_fg.current_estimate()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# origins = np.array([(node).translation() for node in sim.trajectory.trajectory])

ax.grid()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 3])
ax.set_zlim([-1, 1])


print(result)


graph = slam_fg.graph
FactorGraphVisualization.draw_factor_graph('./Output/', graph, result)
plot_graph3D(graph, result, ax=ax, draw_cov=False)
plt.show()
