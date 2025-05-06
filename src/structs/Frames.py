import cv2
import numpy as np

from copy import copy
from src.util import DataAssociation

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.structs import Map, MapPoint, Camera
    from src.util import Geometry

# https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/include/Frame.h
# https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/src/Frame.cc


class Common:
    """Common functionality between KeyFrame and Frame"""
    Tcw: 'Geometry.SE3'

    def set_pose(self, Tcw: 'Geometry.SE3') -> None: self.Tcw = Tcw
    def get_pose(self) -> 'Geometry.SE3': return self.Tcw
    def get_pose_inverse(self) -> 'Geometry.SE3': return self.Tcw.inverse()
    def get_rotation_inverse(self) -> 'Geometry.SO3': return self.get_pose_inverse().rotation()
    def get_camera_center(self) -> 'Geometry.Vector3': return self.get_pose_inverse().translation()



class Frame(Common):
    next_id = 0 # Class shared variable

    def __init__(self, image: 'cv2.Mat', timestep: int, camera: 'Camera') -> None:
        self.image = image
        self.timestep = timestep
        self.camera = camera

        self.id = Frame.next_id
        Frame.next_id += 1 # Update the class shared counter

        # Initialize instance variables
        self.Tcw: 'Geometry.SE3' | None = None
        self.reference_keyframe: 'KeyFrame' | None = None
        self.map_points: dict['MapPoint'] = {}
        self.outlier = False

        self.extract_features()
        self.undistort_keypoints()


    @staticmethod
    def copy(frame: 'Frame') -> 'Frame':
        """Return a copy of the provided frame"""
        return copy(frame)

    def extract_features(self) -> None:
        keypoints, descriptors = DataAssociation.Extractor.extract(self.image, None)
        self.keypoints: list[cv2.KeyPoint] = keypoints # List of keypoints
        self.descriptors: list[cv2.Mat] = descriptors # List of descriptors

    def undistort_keypoints(self) -> None:
        kps = np.array([kp.pt for kp in self.keypoints])
        kps_und = cv2.undistortPoints(kps, self.camera.K, np.array(self.camera.dist_coeffs))
        kps_und = kps_und[:, 0, :]

        self.keypoints_und: list[cv2.KeyPoint] = [] # Undistorted keypoints
        for i, kp in enumerate(self.keypoints):
            x, y = kps_und[i, :]
            kp_result = cv2.KeyPoint(x, y, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
            self.keypoints_und.append(kp_result)



class KeyFrame(Common):
    next_id = 0
    
    def __init__(self, frame: 'Frame' = None, map: 'Map' = None) -> None:
        
        self.frame_id = 0 if not frame else frame.id
        self.timestep = 0 if not frame else frame.timestep

        self.map_points = {} if not frame else frame.map_points
        self.origin_map_id = 0 if not map else map.get_id()
        
        self.update_map(map)
        self.set_pose(frame.get_pose())

        self.id = KeyFrame.next_id
        KeyFrame.next_id += 1
    
    def get_map(self) -> 'Map': return self.map
    
    def update_map(self, map: 'map') -> None: self.map = map

    def add_map_point(self, map_point: 'MapPoint', id: int) -> None: self.map_points[id] = map_point