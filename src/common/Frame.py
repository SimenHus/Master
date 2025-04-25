
import numpy as np
from cv2 import KeyPoint
from dataclasses import dataclass

@dataclass
class Frame:
    image: np.ndarray
    id: int = -1
    keypoints: tuple[KeyPoint] | None = None
    descriptors: np.ndarray | None = None

    def set_features(self, keypoints, descriptors) -> None:
        self.keypoints = keypoints
        self.descriptors = descriptors

    @property
    def features(self) -> list:
        return self.keypoints, self.descriptors