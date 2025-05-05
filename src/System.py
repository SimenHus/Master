

from src.frontend import Tracker
# from src.mapping import LocalMapping
# from src.backend import LoopClosing

from src.structs import Frame, MapPoint
from src.util import Geometry
import cv2


# See https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/src/System.cc
# and https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/include/System.h#L253
class System:
    # Tracking states
    tracking_state: int
    tracked_map_points: list[MapPoint] # Tracked map points
    tracked_key_points_und: list[cv2.KeyPoint] # Undistorted tracked keypoints


    def __init__(self) -> None:
        self.tracker = Tracker()
        # self.local_mapper = LocalMapping()
        # self.loop_closer = LoopClosing()

    def track_monocular(self, image: cv2.Mat, timestep: int) -> Geometry.SE3:
        """Start of SLAM pipeline, incoming frames are sent here"""
        # Perform checks / changes with frame
        Tcw: Geometry.SE3 = self.tracker.grab_image_monocular(image, timestep)

        self.tracking_state = self.tracker.state
        self.tracked_map_points = self.tracker.current_frame.map_points
        self.tracked_key_points_und = self.tracker.current_frame.keypoints_und

        return Tcw