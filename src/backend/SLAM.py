
from gtsam import ISAM2Params, ISAM2
from gtsam import NonlinearFactorGraph, Values, Symbol
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

        prior = Pose3()
        prior_noise = noiseModel.Diagonal.Sigmas([0.3, 0.3, 0.3, 0.1, 0.1, 0.1])
        self.graph.push_back(PriorFactorPose3(X(1), prior, prior_noise)) # Prior on x1

        parameters = ISAM2Params()
        parameters.setRelinearizeThreshold(0.1)
        parameters.relinearizeSkip = 1
        self.isam = ISAM2(parameters)


    def odometry_measurement(self, from_id: int, to_id: int, odometry: Pose3, odometry_noise: noiseModel) -> None:
        self.graph.push_back(BetweenFactorPose3(X(from_id), X(to_id), odometry, odometry_noise))
    
    def pose_measurement(self, pose_id: int, measurement: Pose3, meaurement_noise: noiseModel) -> None:
        self.graph.push_back(PriorFactorPose3(X(pose_id), measurement, meaurement_noise))
        self.new_nodes.insert(X(pose_id), measurement)


    def landmark_measurement(self, pose_id: int, landmark_id: int, landmark_position_estimate: Point3, pixels: tuple, measurement_noise: noiseModel) -> None:
        self.graph.push_back(CameraExtrinsicFactorAlt(T(1), X(pose_id), L(landmark_id), pixels, measurement_noise))
        node_exists = self.check_node_exists(L(landmark_id))
        if node_exists: return
        self.new_nodes.insert(L(landmark_id), landmark_position_estimate)


    def check_node_exists(self, node_id: Symbol) -> bool:
        in_isam = self.isam.valueExists(node_id)
        in_new_nodes = self.new_nodes.exists(node_id)
        return in_isam or in_new_nodes


    def optimize(self) -> None:
        # Perform incremental update to iSAM
        self.isam.update(self.graph, self.new_nodes)
        self.new_nodes.clear()
        extra_updates = 0
        for _ in range(extra_updates): self.isam.update()


    def current_estimate(self) -> Values:
        return self.isam.calculateBestEstimate()