import cv2
import numpy as np

from copy import copy
from src.structs import MapPointDB

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.structs import Map, MapPoint, Camera
    from src.util import Geometry, DataAssociation

# https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/include/Frame.h
# https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/src/Frame.cc


class Common:
    """Common functionality between KeyFrame and Frame"""
    _Tcw: 'Geometry.SE3'
    _map_points: 'MapPointDB'

    def set_pose(self, Tcw: 'Geometry.SE3') -> None: self._Tcw = Tcw
    def get_pose(self) -> 'Geometry.SE3': return self._Tcw
    def get_pose_inverse(self) -> 'Geometry.SE3': return self._Tcw.inverse()
    def get_rotation_inverse(self) -> 'Geometry.SO3': return self.get_pose_inverse().rotation()
    def get_camera_center(self) -> 'Geometry.Vector3': return self.get_pose_inverse().translation()

    def clear_map_points(self) -> None: self._map_points = MapPointDB()
    def get_map_points(self) -> 'MapPointDB': return self._map_points
    def add_map_point(self, map_point: 'MapPoint', id: int) -> None: self.get_map_points()[id] = map_point
    def update_map_point(self, map_point: 'MapPoint') -> None: self.get_map_points().update_map_point(map_point)
    def set_map_points(self, map_points: list['MapPoint'], ids: list[int]) -> None:
        self.clear_map_points()
        for map_point, id in zip(map_points, ids): self.add_map_point(map_point, id)

class Frame(Common):
    next_id = 0 # Class shared variable

    def __init__(self, image: 'cv2.Mat', timestep: int, extractor: 'DataAssociation.Extractor', camera: 'Camera') -> None:
        self.image = image
        self.timestep = timestep
        self.camera = camera
        self.extractor = extractor

        self.id = Frame.next_id
        Frame.next_id += 1 # Update the class shared counter

        # Initialize instance variables
        self._Tcw: 'Geometry.SE3' | None = None
        self.reference_keyframe: 'KeyFrame' | None = None
        self._map_points = MapPointDB()

        self.scale_factors = self.extractor.get_scale_factors()
        self.extract_features()
        self.undistort_keypoints()

    def get_outlier_ids(self) -> set[int]: return self.get_map_points().get_outlier_ids()

    def add_outlier_id(self, id: int) -> None: self.get_map_points().set_outlier(id)

    def get_outliers(self) -> 'MapPointDB':
        """Get map points marked as outliers"""
        return self.get_map_points().get_outliers()

    @staticmethod
    def clone(frame: 'Frame') -> 'Frame':
        """Return a copy of the provided frame"""
        return copy(frame)

    def extract_features(self) -> None:
        keypoints, descriptors = self.extractor.extract(self.image, None)
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
        self._map_points = MapPointDB() if not frame else frame._map_points
        self.scale_factors = [] if not frame else frame.scale_factors
        self.keypoints: list[cv2.KeyPoint] = [] if not frame else frame.keypoints
        self.descriptors: list[cv2.Mat] = [] if not frame else copy(frame.descriptors) # Copy to make sure we do not alter the frame descriptors
        self.keypoints_und: list[cv2.KeyPoint] = [] if not frame else frame.keypoints_und

        self.origin_map_id = 0 if not map else map.get_id()
        
        self.update_map(map)
        self.set_pose(frame.get_pose())

        self.id = KeyFrame.next_id
        KeyFrame.next_id += 1
    
    def get_map(self) -> 'Map': return self._map
    
    def update_map(self, map: 'map') -> None: self._map = map

    def compute_scene_median_depth(self) -> float:
        if len(self.keypoints) == 0: return -1.0

        map_points = self.get_map_points()
        Tcw = self.get_pose()

        # Get a list of z coordinates in camera frame for each map point
        depths = np.array([Tcw.transformTo(map_point.get_world_pos())[2] for map_point in map_points])
        
        # Return median of sorted list of depths (z coordinates)
        return np.median(np.sort(depths))


    def as_dict(self) -> dict:
        return {
            'id': self.id,
            'timestep': self.timestep,
            'keypoints': [kp.pt for kp in self.keypoints],
            'Twc': self.get_pose_inverse().matrix().tolist()
        }