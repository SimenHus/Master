
from gtsam import ISAM2Params, ISAM2, Marginals
from gtsam import NonlinearFactorGraph, Values, Symbol
from gtsam import PriorFactorPose3, BetweenFactorPose3, BearingRangeFactor3D, PriorFactorPoint3, Pose3, Point3, Rot3, GPSFactor
from gtsam import noiseModel, LevenbergMarquardtOptimizer, DoglegOptimizer
from gtsam.symbol_shorthand import X, L, T
import gtsam

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from Visualization.GraphVisualization import FactorGraphVisualization
from Visualization.PlotVisualization import plot_graph3D

from Simulation.factor import CameraExtrinsicFactorAlt

graph = NonlinearFactorGraph()
initial = Values()

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

prior = Pose3()
prior_noise = noiseModel.Diagonal.Sigmas([0.3, 0.3, 0.3, 0.1, 0.1, 0.1])

graph.push_back(PriorFactorPose3(X(1), prior, prior_noise)) # Prior on x1
graph.push_back(BetweenFactorPose3(X(1), X(2), odometry, odometry_noise))
graph.push_back(BetweenFactorPose3(X(2), X(3), odometry, odometry_noise))


for i in range(len(position_gt)):
    pos = position_gt[i] + position_noisy[i] # Include noise in inital estimates
    rot = rotation_gt[i]
    # graph.push_back(GPSFactor(X(i+1), position_gt[i], position_noise)) # Add GNSS measurements
    graph.push_back(PriorFactorPose3(X(i+1), Pose3(rot, position_gt[i]), pose_noise))
    initial.insert(X(i+1), Pose3(rot, pos)) # Add inital estimates

camera_pos = Point3(1, 1, 1)
camera_rot = Rot3(1, 0, 0, 0)
camera_extrinsics = Pose3(camera_rot, camera_pos)
camera_prior = Pose3(Rot3(1, 0, 0, 0), Point3(0.8, 1.2, 1.2))
# camera_prior = Pose3()
# camera_prior = camera_extrinsics
sigma = 1e6
camera_prior_noise = noiseModel.Diagonal.Sigmas([sigma]*6)

# graph.push_back(PriorFactorPose3(T(1), camera_prior, camera_prior_noise))
initial.insert(T(1), camera_prior)

extrinsic_factor_noise = noiseModel.Diagonal.Sigmas([0.1, 0.1, 0.1])
graph.push_back(CameraExtrinsicFactorAlt(T(1), X(1), L(1), (Pose3(rotation_gt[0], position_gt[0]).compose(camera_extrinsics)).transformTo(landmarks_world[0]), extrinsic_factor_noise))
graph.push_back(CameraExtrinsicFactorAlt(T(1), X(2), L(1), (Pose3(rotation_gt[1], position_gt[1]).compose(camera_extrinsics)).transformTo(landmarks_world[0]), extrinsic_factor_noise))
graph.push_back(CameraExtrinsicFactorAlt(T(1), X(2), L(2), (Pose3(rotation_gt[1], position_gt[1]).compose(camera_extrinsics)).transformTo(landmarks_world[1]), extrinsic_factor_noise))
graph.push_back(CameraExtrinsicFactorAlt(T(1), X(3), L(2), (Pose3(rotation_gt[2], position_gt[2]).compose(camera_extrinsics)).transformTo(landmarks_world[1]), extrinsic_factor_noise))

initial.insert(L(1), landmarks_world[0])
initial.insert(L(2), landmarks_world[1])
        

result = LevenbergMarquardtOptimizer(graph, initial).optimize()

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

# gaussian_graph = graph.linearize(result)
# info_matrix, info_vector = gaussian_graph.hessian()
# print(info_matrix.shape)


FactorGraphVisualization.draw_factor_graph('./Output/', graph, result)
plot_graph3D(graph, result, ax=ax, draw_cov=False)
plt.show()
