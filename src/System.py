

from src.frontend import Tracker
from src.atlas import Atlas
import json
# from src.mapping import LocalMapping
# from src.backend import LoopClosing

from src.util import Geometry, Logging
import cv2

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.structs import MapPoint

# See https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/src/System.cc
# and https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/include/System.h#L253
class System:
    # Tracking states
    tracking_state: int
    tracked_map_points: set['MapPoint'] # Tracked map points
    tracked_key_points_und: list[cv2.KeyPoint] # Undistorted tracked keypoints


    def __init__(self) -> None:
        self.atlas = Atlas()
        self.tracker = Tracker(self.atlas)
        # self.local_mapper = LocalMapping()
        # self.loop_closer = LoopClosing()

        Logging.setup_logging('./debug')

    def track_monocular(self, image: cv2.Mat, timestep: int, mask = None) -> Geometry.SE3:
        """Start of SLAM pipeline, incoming frames are sent here"""
        # Perform checks / changes with frame
        Tcw: Geometry.SE3 = self.tracker.grab_image_monocular(image, timestep, mask)

        self.tracking_state = self.tracker.state
        self.tracked_map_points = self.tracker.current_frame.get_map_points()
        self.tracked_key_points_und = self.tracker.current_frame.keypoints_und

        return Tcw
    
    def save_keyframes(self, filename: 'str') -> None:
        keyframes = self.atlas.get_all_keyframes()
        keyframes = sorted(keyframes, key=lambda kf: kf.id)
        json_dict = {}
        for keyframe in keyframes:
            json_dict[keyframe.id] = keyframe.as_dict()


        json_str = json.dumps(json_dict, indent=4)
        with open(f'{filename}.json', 'w') as f:
            f.write(json_str)

    def save_map_points(self, filename: 'str') -> None:

        map_points = self.atlas.get_all_map_points()
        
        json_dict = {}
        for map_point in map_points:
            json_dict[map_point.id] = map_point.as_dict()

        json_str = json.dumps(json_dict, indent=4)
        with open(f'{filename}.json', 'w') as f:
            f.write(json_str)