

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

    def __init__(self, parameters: list) -> None:
        self.parameters = parameters # [fx, fy, cx, cy]

        self.id = Camera.next_id # Get ID
        Camera.next_id += 1 # Update global ID counter
        

    @property
    def K(self) -> np.ndarray[3, 3]:
        return np.array([
            [self.parameters[0], 0, self.parameters[2]],
            [0, self.parameters[1], self.parameters[3]],
            [0, 0, 1]
        ])

    def reconstruct_with_two_views(self, kps1: list[cv2.KeyPoint], kps2: list[cv2.KeyPoint], matches12: list[cv2.DMatch]) -> tuple[bool, Geometry.SE3, list[Geometry.Point3]]:
        """Function to reconstruct two view geometry"""
        # Define lists of matching points from the two images
        im1_kps = [kps1[match.queryIdx] for match in matches12]
        im2_kps = [kps2[match.trainIdx] for match in matches12]

        E = cv2.findEssentialMat(im1_kps, im2_kps, self.K, cv2.RANSAC, 0.999, 1.0) # Get essential matrix using same points in the two images
        retval, R, t, mask, triangulated_points = cv2.recoverPose(E, im1_kps, im2_kps, self.K) # Recover pose from the two images using the calculated essential matrix

        T21 = Geometry.SE3(R, t)

        return True, T21, triangulated_points
