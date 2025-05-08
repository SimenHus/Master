

from src.frontend import Tracker
from src.atlas import Atlas
import pickle
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

        Logging.setup_logging()

    def track_monocular(self, image: cv2.Mat, timestep: int) -> Geometry.SE3:
        """Start of SLAM pipeline, incoming frames are sent here"""
        # Perform checks / changes with frame
        Tcw: Geometry.SE3 = self.tracker.grab_image_monocular(image, timestep)

        self.tracking_state = self.tracker.state
        self.tracked_map_points = self.tracker.current_frame.get_map_points()
        self.tracked_key_points_und = self.tracker.current_frame.keypoints_und

        return Tcw
    
    def save_keyframe_trajectory(self, filename: 'str') -> None:
        keyframes = self.atlas.get_all_keyframes()

        sorted_keyframes_list = sorted(keyframes, key=lambda kf: kf.timestep)

        with open(f'{filename}.txt', 'w') as f:
            f.write('Timestep ; Rotation [w, x, y, z] ; Translation [x, y, z]')
            for keyframe in sorted_keyframes_list:
                Twc = keyframe.get_pose_inverse()
                q = Twc.rotation().toQuaternion()
                t = Twc.translation()
                f.write(f'\n{keyframe.timestep};{q.coeffs()};{t}')


    def save_map_points(self, filename: 'str') -> None:

        with open(filename, 'ab') as f:
            pickle.dump(self.atlas.get_all_map_points(), f)