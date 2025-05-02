import numpy as np

from src.camera import Frame, Camera, KeyFrame
from src.measurements import CameraMeasurement
from src.motion import Pose3

from dataclasses import dataclass, field
from src.camera import Keypoint, Descriptor

@dataclass
class MapPointObservation:
    timestep: int
    camera_id: int
    keypoint: Keypoint = None

    def __eq__(self, other) -> bool:
        if not isinstance(other, MapPointObservation): return False
        return self.timestep == other.timestep and self.camera_id == other.timestep
    
    def __hash__(self) -> int:
        return hash(tuple(self.timestep, self.camera_id))

@dataclass
class MapPoint:
    id: int
    position: np.ndarray[3]
    descriptor: Descriptor
    observations: set[MapPointObservation] = field(default_factory=set)
    # visible_times = 0
    # matched_times = 0
    # normal_vector = np.zeros(3)
    is_outlier = False

    def add_observation(self, frame: Frame, keypoint: Keypoint):
        self.observations.add(MapPointObservation(frame.timestep, frame.camera_id, keypoint))

    # def compute_descriptor(self, descriptors):
    #     """Average or median of descriptors from all observations"""
    #     self.descriptor = np.mean(descriptors, axis=0)

    # def update_normal(self, frame_pose):
    #     """Update normal vector based on camera viewing direction"""
    #     view_dir = (self.position - frame_pose[:3, 3])
    #     view_dir /= np.linalg.norm(view_dir)
    #     self.normal_vector += view_dir
    #     self.normal_vector /= np.linalg.norm(self.normal_vector)

    def __eq__(self, other) -> bool:
        if not isinstance(other, MapPoint): return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        return self.id



# See https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/src/KeyFrameDatabase.cc
class KeyFrameDatabase:
    map_points: set[MapPoint] = set()
    keyframes: set[KeyFrame] = set()
    next_id = 1

    def add_map_point(self, descriptor: Descriptor, position: np.ndarray[3], observation: MapPointObservation) -> None:
        """Add map point to map"""
        point = MapPoint(self.next_id, position, descriptor)
        point.add_observation(observation)
        self.next_id += 1

    def remove_map_point(self, id: int) -> None:
        if id in self.map_points: self.map_points.remove(id)

    def get_map_points(self) -> set[MapPoint]:
        """Get all map points, excluding outliers"""
        return {mpt for mpt in self.map_points if not mpt.is_outlier}
    
    def cull_outliers(self, visibility_treshold: float = 2, match_ratio_threshold: float = 0.25) -> None:
        # for mp_id, mp in list(self.map_points.items()):
        #     if (mp.visible_times < visibility_thresh or
        #         mp.matched_times / (mp.visible_times + 1e-6) < match_ratio_thresh):
        #         mp.is_outlier = True
        pass


    def add(self, frame: Frame, camera_pose: Pose3) -> None:
        """Add frame as keyframe to map"""
        keyframe = KeyFrame(frame, camera_pose)
        self.keyframes.add(keyframe)

    def erase(self): pass

    def detect_loop_candidates(self): pass
