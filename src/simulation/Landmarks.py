
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.util import Geometry


class LandmarkGenerator:

    @classmethod
    def grid_mps(clc) -> list['Geometry.Point3']:
        mps = [
            np.array([10, 10, 10]),
            np.array([-10, 10, 10]),
            np.array([10, -10, 10]),
            np.array([-10, -10, 10]),
            np.array([10, 10, -10]),
            np.array([-10, 10, -10]),
            np.array([10, -10, -10]),
            np.array([-10, -10, -10]),
            np.array([5, 5, 5]),
            np.array([-5, 5, 5]),
            np.array([5, -5, 5]),
            np.array([-5, -5, 5]),
            np.array([5, 5, -5]),
            np.array([-5, 5, -5]),
            np.array([5, -5, -5]),
            np.array([-5, -5, -5]),
        ]
        return mps