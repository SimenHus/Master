

import cv2
from dataclasses import dataclass

from src.util import Geometry

@dataclass
class ImageData:
    image: cv2.Mat
    timestep: str
    filename: str

@dataclass
class LLA:
    lat: float
    lon: float
    alt: float


@dataclass
class STXData:
    lla: LLA
    att: Geometry.Vector3
    timestep: str
