import numpy as np

from .RigidMotion import SE3, SE2


class PositionParameter:
    def __init__(self, nonzero: bool = True, min: int = -2, max: int = 2) -> None:
        self.nonzero = nonzero
        self.min = min
        self.max = max


class RotationParameter:
    def __init__(self, nonzero: bool = True, min: int = -2, max: int = 2) -> None:
        self.nonzero = nonzero
        self.min = min
        self.max = max
   

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
                    r[j] = self.RNG.integers(low=param.min, high=param.max).astype(np.float64)
            R = self.group.rot.Expmap(r)
            odom = self.group.pose(R, t)
            node = self.trajectory[i]*odom

            self.trajectory.append(node)
            self.odometry.append(odom)

class TrajectoryPlanar3D(TrajectoryGeneratorBaseClass):
    group = SE3
    x_params = PositionParameter(nonzero=True)
    y_params = PositionParameter(nonzero=True)
    z_params = PositionParameter(nonzero=False)
    roll_params = RotationParameter(nonzero=False)
    pitch_params = RotationParameter(nonzero=False)
    yaw_params = RotationParameter(nonzero=False)

    position_parameters = [x_params, y_params, z_params]
    rotation_parameters = [roll_params, pitch_params, yaw_params]



class TrajectoryPlanar2D(TrajectoryGeneratorBaseClass):
    group = SE2
    x_params = PositionParameter(nonzero=True)
    y_params = PositionParameter(nonzero=True)
    roll_params = RotationParameter(nonzero=False)

    position_parameters = [x_params, y_params]
    rotation_parameters = [roll_params]
