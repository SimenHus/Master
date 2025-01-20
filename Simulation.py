import numpy as np
from gtsam import Pose3, Point3, Rot3
from gtsam import Pose2, Point2, Rot2


class TrajectoryGeneratorBaseClass:
    odometry: list
    trajectory: list


class Planar3D(TrajectoryGeneratorBaseClass):
    
    def __init__(self, n_nodes: int, prior: Pose3 = None) -> None:
        prior = prior if prior is not None else Pose3()
        self.n_nodes = n_nodes
        self.trajectory = [prior] # Every node in the trajectory
        self.odometry = [] # Odometry between nodes
        self.RNG = np.random.default_rng()


    def generate_trajectory(self) -> None:
        for i in range(self.n_nodes):
            x, y = self.RNG.integers(low=-2, high=2, size=2).astype(np.float64)
            t = Point3(x, y, 0.0)
            R = Rot3()
            odom = Pose3(R, t)
            node = self.trajectory[i]*odom

            self.trajectory.append(node)
            self.odometry.append(odom)


class Planar2D(TrajectoryGeneratorBaseClass):
    
    def __init__(self, n_nodes: int, prior: Pose2 = None) -> None:
        prior = prior if prior is not None else Pose2()
        self.n_nodes = n_nodes
        self.trajectory = [prior] # Every node in the trajectory
        self.odometry = [] # Odometry between nodes
        self.RNG = np.random.default_rng()
        self.generate_trajectory()

    def generate_trajectory(self) -> None:
        for i in range(self.n_nodes):
            x, y = self.RNG.integers(low=-2, high=2, size=2).astype(np.float64)
            t = Point2(x, y)
            R = Rot2()
            odom = Pose2(R, t)
            node = self.trajectory[i]*odom

            self.trajectory.append(node)
            self.odometry.append(odom)