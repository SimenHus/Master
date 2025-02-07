import numpy as np
from ..util.RigidMotion import SE3, SE2


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