
from gtsam import ISAM2Params, ISAM2
from gtsam import NonlinearFactorGraph, Values
from gtsam import PriorFactorPose3, BetweenFactorPose3, Pose3, Point3
from gtsam import noiseModel
from gtsam.symbol_shorthand import X, L, T

from Simulation.factor import CameraExtrinsicFactorAlt

class SLAM:
    def __init__(self) -> None:
        self.graph = NonlinearFactorGraph()
        self.new_nodes = Values()

        camera_initial_guess = Pose3()
        camera_prior = Pose3()
        sigma = 1e6
        camera_prior_noise = noiseModel.Diagonal.Sigmas([sigma]*6)
        self.new_nodes.insert(T(1), camera_initial_guess)
        self.graph.push_back(PriorFactorPose3(T(1), camera_prior, camera_prior_noise))

        self.xn = 1 # Current pose iteration

        parameters = ISAM2Params()
        parameters.setRelinearizeThreshold(0.1)
        parameters.relinearizeSkip = 1
        self.isam = ISAM2(parameters)
    
    def pose_measurement(self, odometry: Pose3, measurement: Pose3, odometry_noise: noiseModel, meaurement_noise: noiseModel) -> None:
        self.xn += 1 # Increment current pose step

        self.graph.push_back(PriorFactorPose3(X(self.xn), measurement, meaurement_noise))
        self.graph.push_back(BetweenFactorPose3(X(self.xn-1), X(self.xn), odometry, odometry_noise))

        self.new_nodes.insert(X(self.xn), measurement)


    def landmark_measurement(self, landmark_id: int, landmark_position_estimate: Point3, pixels: tuple, measurement_noise: noiseModel) -> None:
        self.graph.push_back(CameraExtrinsicFactorAlt(T(1), X(self.xn), L(landmark_id), pixels, measurement_noise))

        self.new_nodes.insert(L(landmark_id), landmark_position_estimate)


    def include_new_nodes(self) -> None:
        # Perform incremental update to iSAM
        self.isam.update(self.graph, self.new_nodes)
        self.new_nodes.clear()
        extra_updates = 0
        for _ in range(extra_updates): self.isam.update()


    def current_estimate(self) -> Values:
        return self.isam.calculateBestEstimate()