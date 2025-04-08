
import numpy as np
from dataclasses import dataclass

@dataclass
class Frame:
    id: int
    image: np.ndarray
    keypoints: tuple | None = None
    descriptors: np.ndarray | None = None