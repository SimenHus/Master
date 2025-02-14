
from .common import *

class CirclePlanar3D:
    odometry: list
    trajectory: list
    group = SE3

    def __init__(self, n_nodes: int, prior = None) -> None:
        self.prior = prior if prior is not None else self.group.pose()
        self.n_nodes = n_nodes
        self.trajectory = [self.prior] # Nodes in the trajectory
        self.odometry = [] # Odometry between nodes
        self.generate_trajectory()

    def generate_trajectory(self) -> None:
        for i in range(self.n_nodes):
            # t = np.zeros(self.group.dim_t) # Define initial zero vector for positional odometry

            t = np.array([1, 0, 0])
            t = self.group.point(t) # Convert position vector to respective Point3 | Point2

            # r = np.zeros(self.group.dim_r) # Define initial zero vector for rotational odometry
            r = np.array([0, 0, 0.1])
            R = self.group.rot.Expmap(r)

            odom = self.group.pose(R, t)
            node = self.trajectory[i].compose(odom)

            self.trajectory.append(node)
            self.odometry.append(odom)