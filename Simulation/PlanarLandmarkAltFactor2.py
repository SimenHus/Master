
from .common import *
from .BaseClass import SimulationBaseClass

class PlanarLandmarkAltFactor2(SimulationBaseClass):
    def __init__(self, steps: int = 10, extrinsic_prior = None, extrinsic_cov = None) -> None:
        # Define ISAM2 parameters
        parameters = ISAM2Params()
        parameters.setRelinearizeThreshold(0.1)
        parameters.relinearizeSkip = 1

        self.inital_estimate = Values()

        self.trajectory = CirclePlanar3D(steps)

        self.odometry_noise = Odometry3D
        self.odometry_factor = BetweenFactorPose3

        # print(Symbol('x', 10).value())

        self.measurement_noise = Odometry3D

        self.landmark_noise = Special
        self.landmark_factor = CameraExtrinsicFactorAlt

        self.prior_noise = self.odometry_noise
        self.prior_factor = PriorFactorPose3

        # quat = [0, 0, 0, 1]
        quat = [1, 0, 0, 0]
        rot = Rot3(*quat)
        t = Point3(0, 3, 0)
        sigma = 1e6
        self.camera_extrinsics = Pose3(rot, t)
        self.camera_extrinsics_covariance = noiseModel.Diagonal.Sigmas([sigma]*6) if extrinsic_cov is None else extrinsic_cov
        # camera_prior = extrinsic_prior if extrinsic_prior is not None else self.camera_extrinsics
        camera_prior = Pose3()
        camera_prior = SimulatedMeasurement.sample(self.camera_extrinsics, self.measurement_noise)
        camera_prior = self.camera_extrinsics


        self.steps = steps
        self.current_step = 0

        self.ISAM = ISAM2(parameters) # Set ISAM2 parameters

        self.graph = NonlinearFactorGraph() # Define the factorgraph
        self.current_estimate = Values() # Current estimate from ISAM

        self.prior_index = 0 # Index for prior
        self.graph.push_back(self.prior_factor(T(0), camera_prior, self.camera_extrinsics_covariance))
        self.inital_estimate.insert(T(0), camera_prior) # Camera extrinsic prior

        # self.graph.push_back(self.prior_factor(X(1), self.trajectory.prior, self.prior_noise.default_noise_model()))

        dist = 5
        self.landmark_index = 0
        self.landmarks = [Point3(dist, 0, 0), Point3(-dist, 0, 0), Point3(0, dist, 0), Point3(0, -dist, 0)]
        self.landmarks = [Point3(0, dist, 0), Point3(dist, 0, 0)]
        self.graph.push_back(PriorFactorPoint3(L(0), self.landmarks[0], self.landmark_noise.default_noise_model()))
        for i, landmark in enumerate(self.landmarks):
            self.inital_estimate.insert(L(self.landmark_index + i), landmark)

    def sim_step(self, i: int) -> None:
        self.current_step += 1
        key = i + self.prior_index + 1 # Key for current factor

        base_true_odom = self.trajectory.odometry[i] # Current true odometry
        base_true_state = self.trajectory.trajectory[i + 1] # Current true state
        camera_true_state = base_true_state.compose(self.camera_extrinsics)

        odometry_measurement = SimulatedMeasurement.sample(base_true_odom, self.odometry_noise) # Sample noisy odometry measurement
        camera_measurement = SimulatedMeasurement.sample(camera_true_state, self.measurement_noise) # Sample noisy camera measurement
        reference_frame_measurement = SimulatedMeasurement.sample(base_true_state, self.measurement_noise) # Sample noisy reference frame measurement


        for j, landmark in enumerate(self.landmarks):
            landmark_camera = camera_true_state.transformTo(landmark)
            self.graph.push_back(self.landmark_factor(T(0), X(key), L(self.landmark_index + j), landmark_camera, self.landmark_noise.default_noise_model()))


        self.graph.push_back(PriorFactorPose3(X(key), reference_frame_measurement, self.measurement_noise.default_noise_model()))
        if i != 0: self.graph.push_back(BetweenFactorPose3(X(key - 1), X(key), odometry_measurement, self.odometry_noise.default_noise_model()))

        if i == 0:
            # computed_pose = reference_frame_measurement
            computed_pose = base_true_state
        else:
            if i == 1:
                current_pose_estimate = self.inital_estimate.atPose3(X(key-1))
            else:
                current_pose_estimate = self.current_estimate.atPose3(X(key-1))
            computed_pose = current_pose_estimate.compose(odometry_measurement) # Compute estimate of the new state using odometry

        self.inital_estimate.insert(X(key), computed_pose) # Prepare the new state estimate to be added to ISAM

        if i == 0: return
        # Perform incremental update to iSAM
        self.ISAM.update(self.graph, self.inital_estimate)
        extra_updates = 0
        for _ in range(extra_updates): self.ISAM.update()
        self.current_estimate = self.ISAM.calculateEstimate()
        self.inital_estimate.clear()

        marginals = Marginals(self.graph, self.current_estimate)
        # self.camera_extrinsics_covariance = marginals.marginalCovariance(0)