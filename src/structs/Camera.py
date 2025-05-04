

import cv2
import numpy as np

from src.util import Geometry


# class CameraModel:
#     dim: tuple[int, int, int] # Camera dimensions in pixels (height, width, channels)
#     focal_length: tuple[int, int] # Focal length in pixels
#     principal_point: tuple[int, int] # Optical center in pixels
#     distortion_coefficients: tuple # Distortion coefficients
#     skew: float = 0.

#     @property
#     def K(self) -> np.ndarray[3, 3]:
#         return np.array([
#             [self.focal_length[0], self.skew, self.principal_point[0]],
#             [0, self.focal_length[1], self.principal_point[1]],
#             [0, 0, 1]
#         ])
    
#     @property
#     def noise_model(self) -> CameraNoiseModel:
#         return CameraNoiseModel()
    
#     @staticmethod
#     def from_json(json_dict: dict) -> 'CameraModel':
#         K = np.array(json_dict['camera_matrix']).reshape((3, 3))
#         return CameraModel(
#             dim = (2056, 2464, 3),
#             focal_length = (K[0, 0], K[1, 1]),
#             principal_point = (K[0, 2], K[1, 2]),
#             distortion_coefficients = json_dict['distortion_coefficients'],
#             skew = K[0, 1]
#         )


class Camera:
    next_id = 0


    def __init__(self) -> None:
        pass


    def reconstruct_with_two_views(self, keys1: list[cv2.KeyPoint], keys2: list[cv2.KeyPoint], matches12: list[int], T21: Geometry.SE3, P3D: list[Geometry.Point3], triangulated: list[bool]) -> bool:
        pass