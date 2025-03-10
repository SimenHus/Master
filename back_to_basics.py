
from gtsam import ISAM2Params, ISAM2, Marginals
from gtsam import NonlinearFactorGraph, Values, Symbol
from gtsam import PriorFactorPose3, BetweenFactorPose3, BearingRangeFactor3D, PriorFactorPoint3, Pose3, Point3, Rot3
from gtsam import noiseModel, LevenbergMarquardtOptimizer
from gtsam.symbol_shorthand import X, L, T


import matplotlib.pyplot as plt
import numpy as np

from Visualization.GraphVisualization import FactorGraphVisualization
from Visualization.PlotVisualization import plot_graph3D

graph = NonlinearFactorGraph()
initial = Values()

landmarks = [
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

position_noise = [
    Point3(0.3, -0.1, 0.2),
    Point3(-0.2, 0.3, -0.2),
    Point3(0.1, -0.3, -0.3)
]


odometry = Pose3(Rot3(), Point3(0, 1, 0))
odometry_noise = noiseModel.Diagonal.Sigmas([0.2, 0.2, 0.2, 0.1, 0.1, 0.1]) # Odometry noise model

prior = Pose3()
prior_noise = noiseModel.Diagonal.Sigmas([0.3, 0.3, 0.3, 0.1, 0.1, 0.1])

graph.push_back(PriorFactorPose3(X(1), prior, prior_noise)) # Prior on x1
graph.push_back(BetweenFactorPose3(X(1), X(2), odometry, odometry_noise))
graph.push_back(BetweenFactorPose3(X(2), X(3), odometry, odometry_noise))


for i in range(len(position_gt)):
    pos = position_gt[i] + position_noise[i] # Include noise in inital estimates
    rot = rotation_gt[i]
    initial.insert(X(i+1), Pose3(rot, pos)) # Add inital estimates

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

FactorGraphVisualization.draw_factor_graph('./Output/', graph, result)
plot_graph3D(graph, result, ax=ax, draw_cov=True)
plt.show()
