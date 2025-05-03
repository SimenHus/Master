

from src.util import Geometry
from .KeyFrame import KeyFrame
import cv2

from copy import copy

# https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/include/Frame.h
# https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/src/Frame.cc


class Frame:
    next_id = 0 # Class shared variable

    def __init__(self, image: cv2.Mat, timestep: int) -> None:
        self.image = image
        self.timestep = timestep

        self.id = Frame.next_id
        Frame.next_id += 1 # Update the class shared counter

        # Initialize instance variables
        self.Tcw: Geometry.SE3 | None = None
        self.reference_KF: KeyFrame | None = None
        self.keypoints: list[cv2.KeyPoint] = [] # Keypoints
        self.keypoint_und: list[cv2.KeyPoint] = [] # Undistorted keypoints


    @staticmethod
    def copy(frame: 'Frame') -> 'Frame':
        """Return a copy of the provided frame"""
        return copy(frame)

    # def set_features(self, features: list[Feature]) -> None:
    #     self.features = features

    # def extract_features(self) -> None:
    #     keypoints, descriptors = self.orb.detectAndCompute(self.image, None)
    #     features = [Feature(kp, desc) for (kp, desc) in zip(keypoints, descriptors)]
    #     self.set_features(features)

    def set_pose(self, Tcw: Geometry.SE3) -> None:
        self.Tcw = Tcw

    def get_pose(self) -> Geometry.SE3:
        if self.Tcw is None: print('Pose has not been set')
        return self.Tcw

    # @property
    # def keypoints(self) -> list[KeyPoint]:
    #     return [feature.keypoint for feature in self.features]
    
    # @property
    # def descriptors(self) -> list[Descriptor]:
    #     return [feature.descriptor for feature in self.features]
