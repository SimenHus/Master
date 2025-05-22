

import cv2
from dataclasses import dataclass

from src.util import Geometry

@dataclass
class ImageData:
    image: cv2.Mat
    timestep: str
    filename: str


@dataclass
class PoseData:
    pose: Geometry.SE3
    timestep: str