

import cv2
import numpy as np

from src.util import Geometry, Logging

class Camera:
    next_id = 0

    def __init__(self, parameters: list, dist_coeffs: list) -> None:
        self.parameters = parameters # [fx, fy, cx, cy]
        self.dist_coeffs = dist_coeffs
        self.id = Camera.next_id # Get ID
        Camera.next_id += 1 # Update global ID counter
        
        self.logger = Logging.get_logger(f'Camera {self.id}')

    @property
    def parameters_with_skew(self) -> list:
        return [self.parameters[0], self.parameters[1], 0, self.parameters[2], self.parameters[3]]

    @property
    def K(self) -> np.ndarray[3, 3]:
        return np.array([
            [self.parameters[0], 0, self.parameters[2]],
            [0, self.parameters[1], self.parameters[3]],
            [0, 0, 1]
        ])

    def reconstruct_with_two_views(self, kps1: list[cv2.KeyPoint], kps2: list[cv2.KeyPoint], matches12: list[cv2.DMatch]) -> tuple[bool, Geometry.SE3, list[Geometry.Point3], list[bool]]:
        """Function to reconstruct two view geometry"""
        # Define lists of matching points from the two images
        im1_kps = np.array([kps1[match.queryIdx].pt for match in matches12])
        im2_kps = np.array([kps2[match.trainIdx].pt for match in matches12])
        
        dist_thresh = 100.
        E, mask = cv2.findEssentialMat(im1_kps, im2_kps, self.K, cv2.RANSAC, 0.999, 1.0) # Get essential matrix using same points in the two images
        if E is None: return False, None, None, None
        retval, R, t, mask, triangulated_points = cv2.recoverPose(E, im1_kps, im2_kps, self.K, dist_thresh, triangulatedPoints=None, mask=mask) # Recover pose from the two images using the calculated essential matrix

        mask = mask.ravel().astype(bool)
        triangulated_points = np.divide(triangulated_points[:3, :], triangulated_points[3, :]) # Dehomogenize points
        triangulated_points = [point for point in triangulated_points.T] # Convert points to python list of Point3


        R = Geometry.SO3(R)
        T21 = Geometry.SE3(R, t)

        return True, T21, triangulated_points, mask
    
    @staticmethod
    def K_list_to_params(K: list) -> list:
        return [K[0], K[4], K[2], K[5]]
    
    def normed_to_pixels(self, normed_pixels: tuple) -> tuple:
        fx, fy, cx, cy = self.parameters
        u = fx * normed_pixels[0] + cx
        v = fy * normed_pixels[1] + cy
        return (u, v)