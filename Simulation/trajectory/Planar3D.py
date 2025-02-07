from .common import *
from .BaseClass import TrajectoryGeneratorBaseClass

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