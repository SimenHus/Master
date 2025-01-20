from common import *


from Factor import GNSSFactor
from visualization import plot_graph


graph = gtsam.NonlinearFactorGraph()


# Prior

prior_mean = gtsam.Pose2(0.0, 0.0, 0.0)
prior_noise = gtsam.noiseModel.Diagonal.Sigmas([0.3, 0.3, 0.1])
graph.add(gtsam.PriorFactorPose2(1, prior_mean, prior_noise))


# Odometry


odometry = gtsam.Pose2(2.0, 0.0, 0.0)
odometry_noise = gtsam.noiseModel.Diagonal.Sigmas([0.2, 0.2, 0.1])
graph.add(gtsam.BetweenFactorPose2(1, 2, odometry, odometry_noise))
graph.add(gtsam.BetweenFactorPose2(2, 3, odometry, odometry_noise))

# GNSS measurements


GNSS_noise = gtsam.noiseModel.Diagonal.Sigmas((0.1, 0.1))

fac1 = GNSSFactor([0, 0], keys=[1], noise_model=GNSS_noise)
fac2 = GNSSFactor([2, 0], keys=[2], noise_model=GNSS_noise)
fac3 = GNSSFactor([4, 0], keys=[3], noise_model=GNSS_noise)
graph.add(fac1)
graph.add(fac2)
graph.add(fac3)


# Initial values
initial = gtsam.Values()
initial.insert(1, gtsam.Pose2(0.5, 0.0, 0.2))
initial.insert(2, gtsam.Pose2(2.3, 0.1, -0.2))
initial.insert(3, gtsam.Pose2(4.1, 0.1, 0.1))

optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
result = optimizer.optimize()


marginals = gtsam.Marginals(graph, result)

# print('Initial estimate:')
# print(initial)


# print('Optimized estimate:')
# print(result)


# print('Covariance:')
# print(f'x1 cov: \n {marginals.marginalCovariance(1)}')
# print(f'x2 cov: \n {marginals.marginalCovariance(2)}')
# print(f'x3 cov: \n {marginals.marginalCovariance(3)}')




plot_graph(graph, result)