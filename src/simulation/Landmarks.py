
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.util import Geometry


class LandmarkGenerator:

    @staticmethod
    def square(n_points, square_side):
        x = np.random.uniform(-square_side/2, square_side/2, n_points)
        y = np.random.uniform(-square_side/2, square_side/2, n_points)
        z = np.random.uniform(0, 5, n_points)
        return np.column_stack((x, y, z))

    @staticmethod
    def grid_mps() -> list['Geometry.Point3']:
        mps = [
            np.array([20, 20, 5]),
            np.array([-20, 20, 5]),
            np.array([20, -20, 5]),
            np.array([-20, -20, 5]),
            np.array([20, 20, 1]),
            np.array([-20, 20, 1]),
            np.array([20, -20, 1]),
            np.array([-20, -20, 1]),
            np.array([5, 5, 3]),
            np.array([-5, 5, 3]),
            np.array([5, -5, 3]),
            np.array([-5, -5, 3]),
            np.array([5, 5, 1]),
            np.array([-5, 5, 1]),
            np.array([5, -5, 1]),
            np.array([-5, -5, 1]),
        ]
        return mps