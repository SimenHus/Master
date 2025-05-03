
import cv2
from src.structs import Frame
from src.util import Geometry


class Extractor:
    orb = cv2.ORB.create()


class Matcher:
    THRESH_LOW: int = 50
    THRESH_HIGH: int = 100
    HISTO_LENGTH: int = 30
    
    def __init__(self, nnratio: float = 0.6, check_ori: bool = True) -> None:
        self.nnratio = nnratio
        self.check_orientation = check_ori

    def search_for_initialization(self, frame1: Frame, frame2: Frame, prev_matched: list[Geometry.Point2], matches12: list[int]) -> int:
        matches = 0

        # Resize the matches12 reference to same length as nr of undistorted keypoints in frame1
        while len(matches12) > len(frame1.keypoint_und): matches12.pop()
        while len(matches12) < len(frame1.keypoint_und): matches12.append(-1)

        for i in range(len(frame1.keypoint_und)): matches12[i] = -1 # Turn all values in matches12 to -1

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(frame1.descriptors, frame2.descriptors)
        # Return matches sorted by distance (https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)
        return sorted(matches, key=lambda x: x.distance)