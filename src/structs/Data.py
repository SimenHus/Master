

import cv2
from dataclasses import dataclass

from src.util import Geometry

@dataclass
class ImageData:
    image: cv2.Mat
    timestep: str
    filename: str
    mask: cv2.Mat

@dataclass
class LLA:
    lat: float
    lon: float
    alt: float


@dataclass
class STXData:
    state: Geometry.State
    lla: LLA
    timestep: int
