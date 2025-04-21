
import numpy as np
from cv2 import KeyPoint
from dataclasses import dataclass

@dataclass
class Frame:
    id: int
    image: np.ndarray
    keypoints: tuple[KeyPoint] | None = None
    descriptors: np.ndarray | None = None

    def set_features(self, keypoints, descriptors) -> None:
        self.keypoints = keypoints
        self.descriptors = descriptors