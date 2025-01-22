from common import *
from Simulation import Planar2D
from visualization import plot_graph2D
from Sensor import NoisyOdom2D, NoisyMeasurement2D
from Factor import GNSSFactor


ODOMETRY_NOISE = NoisyOdom2D.default_noise_model()
PRIOR_NOISE = NoisyOdom2D.default_noise_model()
MEASUREMENT_NOISE = NoisyMeasurement2D.default_noise_model()


# Create the ground truth odometry measurements of the robot during the trajectory.
trajectory = Planar2D(10)
true_odometry = trajectory.odometry

# Create iSAM2 parameters which can adjust the threshold necessary to force relinearization and how many
# update calls are required to perform the relinearization.
parameters = gtsam.ISAM2Params()
# parameters.setRelinearizeThreshold(0.1)
# parameters.relinearizeSkip = 1
isam = gtsam.ISAM2(parameters)


# Create a Nonlinear factor graph as well as the data structure to hold state estimates.
graph = gtsam.NonlinearFactorGraph()
initial_estimate = gtsam.Values()

# Add the prior factor to the factor graph, and poorly initialize the prior pose to demonstrate
# iSAM2 incremental optimization.
key_start = 1
graph.push_back(gtsam.PriorFactorPose2(key_start, trajectory.prior, PRIOR_NOISE))
initial_estimate.insert(key_start, NoisyOdom2D.sample(trajectory.prior))

# Initialize the current estimate which is used during the incremental inference loop.
current_estimate = initial_estimate


fig, ax = plt.subplots()
origins = np.array([node.translation() for node in trajectory.trajectory])

measurement_sample_rate = 1
for i, true_odom in enumerate(true_odometry):
    key = i + key_start
    # Corrupt the odometry measurements with gaussian noise to create noisy odometry measurements.
    odometry_measurement = NoisyOdom2D.sample(true_odom)

    if i % measurement_sample_rate == 0:
        measurement = NoisyMeasurement2D.sample(trajectory.trajectory[key])
        graph.push_back(GNSSFactor(key + 1, measurement, MEASUREMENT_NOISE))

    # Add a binary factor between a newly observed state and the previous state.
    graph.push_back(gtsam.BetweenFactorPose2(key, key + 1, odometry_measurement, ODOMETRY_NOISE))

    # Compute and insert the initialization estimate for the current pose using the noisy odometry measurement.
    computed_estimate = current_estimate.atPose2(key) * odometry_measurement
    initial_estimate.insert(key + 1, computed_estimate)

    # Perform incremental update to iSAM2's internal Bayes tree, optimizing only the affected variables.
    isam.update(graph, initial_estimate)
    current_estimate = isam.calculateEstimate()

    # Report all current state estimates from the iSAM2 optimzation.
    initial_estimate.clear()

    ax.cla()
    ax.grid()
    plot_graph2D(graph, current_estimate, ax=ax)
    ax.plot(origins[:key+1, 0], origins[:key+1, 1], '-o')
    plt.pause(1)


plt.show()
