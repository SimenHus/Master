from gtsam import ISAM2Params, ISAM2
from gtsam import NonlinearFactorGraph, Values, Symbol
from gtsam import PriorFactorPose3, BetweenFactorPose3
from gtsam import PriorFactorPose2, BetweenFactorPose2


from .Measurements import *
from .TrajectoryGeneration import *
from .Factors import *
from .NoiseModel import *


class SimulationBaseClass:

    def __init__(self, steps: int, ISAM_params) -> None:
        self.steps = steps
        self.current_step = 0

        self.ISAM = ISAM2(ISAM_params) # Set ISAM2 parameters

        self.graph = NonlinearFactorGraph() # Define the factorgraph
        self.current_estimate = Values() # Current estimate from ISAM

        self.key_start = 1 # Where factor keys are initialized

    def simulate_all(self) -> None:
        for i in range(self.steps):
            self.sim_step(i)

    def simulate_step(self) -> None:
        self.sim_step(self.current_step)



class Planar3D(SimulationBaseClass):
    
    def __init__(self, steps: int = 10) -> None:
        # Define ISAM2 parameters
        parameters = ISAM2Params()
        # parameters.setRelinearizeThreshold(0.1)
        # parameters.relinearizeSkip = 1
        super().__init__(steps, parameters)

        self.trajectory = TrajectoryPlanar3D(steps) # Generate trajectory
        self.odometry_noise = Odometry3D.default_noise_model() # Import odometry noise model
        self.measurement_noise = Position3D.default_noise_model() # Import measurement noise model
        self.prior_noise = self.odometry_noise # Define noise on prior estimate

        self.measurement_sample_rate = 1

        self.graph.push_back(PriorFactorPose3(self.key_start, self.trajectory.prior, self.prior_noise)) # Insert prior into graph
        self.current_estimate.insert(self.key_start, SimulatedMeasurement.sample(self.trajectory.prior, Odometry3D)) # Insert noisy prior into estimates


    
    def sim_step(self, i: int) -> None:
        key = i + self.key_start # Previous factor key

        true_odom = self.trajectory.odometry[i] # True odometry
        true_state = self.trajectory.trajectory[key] # True state

        odometry_measurement = SimulatedMeasurement.sample(true_odom, Odometry3D) # Sample noisy odometry measurement

        if i % self.measurement_sample_rate == 0: # Check if external measurement is available
            measurement = SimulatedMeasurement.sample(true_state, Position3D).translation() # Sample noisy external measurement
            self.graph.push_back(PositionFactor3D(key + 1, measurement, self.measurement_noise)) # Add measurement as factor to graph

        # Add a binary factor between a newly observed state and the previous state.
        self.graph.push_back(BetweenFactorPose3(key, key + 1, odometry_measurement, self.odometry_noise))

        # Compute and insert the initialization estimate for the current pose using the noisy odometry measurement.
        computed_estimate = self.current_estimate.atPose3(key) * odometry_measurement # Compute estimate of the new state using odometry
        
        new_state_estimate = Values() if i > 0 else self.current_estimate
        new_state_estimate.insert(key + 1, computed_estimate) # Prepare the new state estimate to be added to ISAM

        # Perform incremental update to iSAM
        self.ISAM.update(self.graph, new_state_estimate)
        self.current_estimate = self.ISAM.calculateEstimate()

        self.current_step += 1