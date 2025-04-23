

from dataclasses import dataclass
from src.motion import Point3

import numpy as np

@dataclass
class LandmarkObservation:
    timestep: int
    pixels: tuple[int, int]

@dataclass
class Landmark:
    id: int
    descriptor: list
    position: np.ndarray[3] = None
    observations: list[LandmarkObservation] = []