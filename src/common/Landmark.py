

from dataclasses import dataclass, field
from src.motion import Point3

import numpy as np

@dataclass
class LandmarkObservation:
    camera_id: int
    timestep: int
    pixels: tuple[int, int]

@dataclass
class Landmark:
    id: int
    descriptor: list
    position: np.ndarray[3] = None
    observations: list[LandmarkObservation] = field(default_factory=list)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Landmark): return False
        # return np.allclose(self.position, other.position, atol=1e-3)
        return tuple(self.descriptor) == tuple(other.descriptor)
    
    def __hash__(self) -> int:
        return hash(tuple(self.position))