
from .common import *

class TrajectoryGeneratorBaseClass:
    odometry: list
    trajectory: list

    def __init__(self, n_nodes: int, prior = None) -> None:
        self.prior = prior if prior is not None else self.group.pose()
        self.n_nodes = n_nodes
        self.trajectory = [self.prior] # Nodes in the trajectory
        self.odometry = [] # Odometry between nodes
        self.RNG = np.random.default_rng()
        self.generate_trajectory()

    def generate_trajectory(self) -> None:
        for i in range(self.n_nodes):
            t = np.zeros(self.group.dim_t) # Define initial zero vector for positional odometry

            for j, param in enumerate(self.position_parameters): # Loop through parameters
                if param.nonzero: # Check if parameter should be nonzero
                    t[j] = self.RNG.integers(low=param.min, high=param.max).astype(np.float64) # Generate random nonzero value
            t = self.group.point(t) # Convert position vector to respective Point3 | Point2

            r = np.zeros(self.group.dim_r) # Define initial zero vector for rotational odometry
            for j, param in enumerate(self.rotation_parameters):
                if param.nonzero:
                    deg = self.RNG.integers(low=param.min, high=param.max).astype(np.float64)
                    r[j] = deg * np.pi / 180
            R = self.group.rot.Expmap(r)
            odom = self.group.pose(R, t)
            node = self.trajectory[i].compose(odom)

            self.trajectory.append(node)
            self.odometry.append(odom)