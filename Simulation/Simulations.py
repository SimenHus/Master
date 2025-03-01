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

        self.prior_index = 0 # Index for prior

        self.graph.push_back(self.prior_factor(self.prior_index, self.trajectory.prior, self.prior_noise.default_noise_model())) # Insert prior into graph
        self.current_estimate.insert(self.prior_index, SimulatedMeasurement.sample(self.trajectory.prior, self.prior_noise)) # Insert noisy prior into estimates


    def simulate_all(self) -> None:
        for i in range(self.steps):
            self.sim_step(i)

    def simulate_step(self) -> None:
        self.sim_step(self.current_step)


    def sim_step(self, i: int) -> None:
        key = i + self.prior_index + 1 # Key for current factor

        true_odom = self.trajectory.odometry[i] # Current true odometry
        true_state = self.trajectory.trajectory[i + 1] # Current true state

        odometry_measurement = SimulatedMeasurement.sample(true_odom, self.odometry_noise) # Sample noisy odometry measurement

        if i % self.measurement_sample_rate == 0: # Check if external measurement is available
            measurement = SimulatedMeasurement.sample(true_state, self.measurement_noise) # Sample noisy external measurement
            self.graph.push_back(self.measurement_factor(key, measurement, self.measurement_noise.default_noise_model())) # Add measurement as factor to graph
        # Add a binary factor between a newly observed state and the previous state.
        self.graph.push_back(self.odometry_factor(key - 1, key, odometry_measurement, self.odometry_noise.default_noise_model()))

        # Compute and insert the initialization estimate for the current pose using the noisy odometry measurement.
        current_estimate = None
        if type(true_odom) == Pose3: current_estimate = self.current_estimate.atPose3(key-1)
        if type(true_odom) == Pose2: current_estimate = self.current_estimate.atPose2(key-1)

        computed_estimate = current_estimate * odometry_measurement # Compute estimate of the new state using odometry
        
        new_state_estimate = Values() if i > 0 else self.current_estimate
        # new_state_estimate = Values()
        new_state_estimate.insert(key, computed_estimate) # Prepare the new state estimate to be added to ISAM

        # Perform incremental update to iSAM
        self.ISAM.update(self.graph, new_state_estimate)
        extra_updates = 0
        for _ in range(extra_updates): self.ISAM.update()
        self.current_estimate = self.ISAM.calculateEstimate()

        self.current_step += 1



class Planar3D(SimulationBaseClass):
    
    def __init__(self, steps: int = 10) -> None:
        # Define ISAM2 parameters
        parameters = ISAM2Params()
        parameters.setRelinearizeThreshold(0.1)
        parameters.relinearizeSkip = 1

        self.trajectory = TrajectoryPlanar3D(steps) # Generate trajectory

        self.odometry_noise = Odometry3D # Import odometry noise model
        self.odometry_factor = BetweenFactorPose3

        self.measurement_sample_rate = 2
        self.measurement_noise = Position3D # Import measurement noise model
        self.measurement_factor = PositionFactor3D # Factor for external measurements

        self.prior_noise = self.odometry_noise # Define noise on prior estimate
        self.prior_factor = PriorFactorPose3


        self.steps = steps
        self.current_step = 0

        self.ISAM = ISAM2(parameters) # Set ISAM2 parameters

        self.graph = NonlinearFactorGraph() # Define the factorgraph
        self.current_estimate = Values() # Current estimate from ISAM

        self.prior_index = 0 # Index for prior

        self.graph.push_back(self.prior_factor(self.prior_index, self.trajectory.prior, self.prior_noise.default_noise_model())) # Insert prior into graph
        self.current_estimate.insert(self.prior_index, SimulatedMeasurement.sample(self.trajectory.prior, self.prior_noise)) # Insert noisy prior into estimates


    def sim_step(self, i: int) -> None:
        key = i + self.prior_index + 1 # Key for current factor

        true_odom = self.trajectory.odometry[i] # Current true odometry
        true_state = self.trajectory.trajectory[i + 1] # Current true state

        odometry_measurement = SimulatedMeasurement.sample(true_odom, self.odometry_noise) # Sample noisy odometry measurement

        if i % self.measurement_sample_rate == 0: # Check if external measurement is available
            measurement = SimulatedMeasurement.sample(true_state, self.measurement_noise) # Sample noisy external measurement
            self.graph.push_back(self.measurement_factor(key, measurement, self.measurement_noise.default_noise_model())) # Add measurement as factor to graph
        # Add a binary factor between a newly observed state and the previous state.
        self.graph.push_back(self.odometry_factor(key - 1, key, odometry_measurement, self.odometry_noise.default_noise_model()))

        # Compute and insert the initialization estimate for the current pose using the noisy odometry measurement.
        current_estimate = None
        if type(true_odom) == Pose3: current_estimate = self.current_estimate.atPose3(key-1)
        if type(true_odom) == Pose2: current_estimate = self.current_estimate.atPose2(key-1)

        computed_estimate = current_estimate * odometry_measurement # Compute estimate of the new state using odometry
        
        new_state_estimate = Values() if i > 0 else self.current_estimate
        # new_state_estimate = Values()
        new_state_estimate.insert(key, computed_estimate) # Prepare the new state estimate to be added to ISAM

        # Perform incremental update to iSAM
        self.ISAM.update(self.graph, new_state_estimate)
        extra_updates = 0
        for _ in range(extra_updates): self.ISAM.update()
        self.current_estimate = self.ISAM.calculateEstimate()

        self.current_step += 1


class Planar2D(SimulationBaseClass):
    
    def __init__(self, steps: int = 10) -> None:
        # Define ISAM2 parameters
        parameters = ISAM2Params()
        parameters.setRelinearizeThreshold(0.1)
        parameters.relinearizeSkip = 1

        self.trajectory = TrajectoryPlanar2D(steps) # Generate trajectory

        self.odometry_noise = Odometry2D # Import odometry noise model
        self.odometry_factor = BetweenFactorPose2

        self.measurement_sample_rate = 1
        self.measurement_noise = Position2D # Import measurement noise model
        self.measurement_factor = PositionFactor2D # Factor for external measurements

        self.prior_noise = self.odometry_noise # Define noise on prior estimate
        self.prior_factor = PriorFactorPose2


        super().__init__(steps, parameters)


    
class Planar3DWithCamera(SimulationBaseClass):
    def __init__(self, steps: int = 10) -> None:
        # Define ISAM2 parameters
        parameters = ISAM2Params()
        parameters.setRelinearizeThreshold(0.1)
        parameters.relinearizeSkip = 1

        self.trajectory = TrajectoryPlanar3D(steps)

        self.odometry_noise = Odometry3D
        self.odometry_factor = BetweenFactorPose3

        self.measurement_sample_rate = 1
        self.measurement_noise = Odometry3D

        self.prior_noise = self.odometry_noise
        self.prior_factor = PriorFactorPose3


        self.camera_extrinsics = Pose3(Rot3(0, 0, 0, 1), Point3(3, 0, 0))


        self.steps = steps
        self.current_step = 0

        self.ISAM = ISAM2(parameters) # Set ISAM2 parameters

        self.graph = NonlinearFactorGraph() # Define the factorgraph
        self.current_estimate = Values() # Current estimate from ISAM

        self.prior_index = 1 # Index for prior

        self.graph.push_back(self.prior_factor(0, self.camera_extrinsics, self.prior_noise.default_noise_model()))
        self.current_estimate.insert(0, SimulatedMeasurement.sample(self.camera_extrinsics, self.prior_noise)) # Camera extrinsic prior

        self.graph.push_back(self.prior_factor(self.prior_index, self.trajectory.prior * self.camera_extrinsics, self.prior_noise.default_noise_model())) # Insert prior into graph
        self.current_estimate.insert(self.prior_index, SimulatedMeasurement.sample(self.trajectory.prior * self.camera_extrinsics, self.prior_noise)) # Insert noisy prior into estimates

    def sim_step(self, i: int) -> None:
        key = i + self.prior_index + 1 # Key for current factor

        true_odom = self.trajectory.odometry[i] # Current true odometry
        true_state = self.trajectory.trajectory[i + 1] # Current true state

        odometry_measurement = SimulatedMeasurement.sample(true_odom * self.camera_extrinsics, self.odometry_noise) # Sample noisy odometry measurement
        camera_measurement = SimulatedMeasurement.sample(true_state * self.camera_extrinsics, self.measurement_noise) # Sample noisy external measurement
        reference_frame_measurement = SimulatedMeasurement.sample(true_state, self.measurement_noise)

        self.graph.push_back(PriorFactorPose3(key, camera_measurement, self.measurement_noise.default_noise_model()))

        self.graph.push_back(CameraExtrinsicFactor3D(0, key, reference_frame_measurement, self.measurement_noise.default_noise_model()))
        self.graph.push_back(BetweenFactorPose3(key - 1, key, odometry_measurement, self.odometry_noise.default_noise_model()))

        # Compute and insert the initialization estimate for the current pose using the noisy odometry measurement.
        current_camera_estimate = self.current_estimate.atPose3(key-1)
        current_extrinsic_estimate = self.current_estimate.atPose3(0)

        computed_camera_estimate = current_camera_estimate * odometry_measurement # Compute estimate of the new state using odometry
        computed_extrinsic_estimate = current_extrinsic_estimate
        
        if i == 0:
            new_state_estimate = self.current_estimate
        else:
            new_state_estimate = Values()

        new_state_estimate.insert(key, computed_camera_estimate) # Prepare the new state estimate to be added to ISAM

        # Perform incremental update to iSAM
        self.ISAM.update(self.graph, new_state_estimate)
        extra_updates = 0
        for _ in range(extra_updates): self.ISAM.update()
        self.current_estimate = self.ISAM.calculateEstimate()

        self.current_step += 1
