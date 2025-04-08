from .common import *
from .BaseClass import TrajectoryGeneratorBaseClass

class TrajectoryPlanar2D(TrajectoryGeneratorBaseClass):
    group = SE2
    x_params = PositionParameter(nonzero=True)
    y_params = PositionParameter(nonzero=True)
    roll_params = RotationParameter(nonzero=False)

    position_parameters = [x_params, y_params]
    rotation_parameters = [roll_params]
