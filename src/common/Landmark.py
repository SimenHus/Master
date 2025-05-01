

from dataclasses import dataclass, field
from src.camera import Keypoint, Descriptor

import numpy as np

@dataclass
class LandmarkObservation:
    camera_id: int
    timestep: int
    keypoint: Keypoint

@dataclass
class Landmark:
    id: int
    descriptor: Descriptor
    position: np.ndarray[3] = None
    observations: list[LandmarkObservation] = field(default_factory=list)

    def add_observation(self, observation: list[LandmarkObservation] | LandmarkObservation) -> None:
        if type(observation) == list: self.observations.extend(observation)
        else: self.observations.append(observation)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Landmark): return False
        # return np.allclose(self.position, other.position, atol=1e-3)
        return tuple(self.descriptor) == tuple(other.descriptor)
    
    def __hash__(self) -> int:
        return hash(tuple(self.descriptor))