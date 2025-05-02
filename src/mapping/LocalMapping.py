import numpy as np

from src.camera import Frame, Camera, KeyFrame
from src.measurements import CameraMeasurement
from src.motion import Pose3

from dataclasses import dataclass, field
from src.camera import Keypoint, Descriptor


class LocalMapping:
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


    def add_keyframe(self, frame: Frame, camera_pose: Pose3) -> None:
        """Add frame as keyframe to map"""
        keyframe = KeyFrame(frame, camera_pose)
        self.keyframes.add(keyframe)
