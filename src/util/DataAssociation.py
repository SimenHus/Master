
import cv2


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.structs import Frame, KeyFrame, MapPoint
    from src.util import Geometry


class Extractor:

    def __init__(self) -> None:
        self.orb = cv2.ORB.create()
        self.scale_factors: list[float] = []
        
    def get_scale_factors(self) -> list[float]: return self.scale_factors

    def extract(self, image, mask) -> tuple[list[cv2.KeyPoint], list[cv2.Mat]]:
        return self.orb.detectAndCompute(image, mask)


class Matcher:
    THRESH_LOW: int = 50
    THRESH_HIGH: int = 100
    HISTO_LENGTH: int = 30
    
    def __init__(self, nnratio: float = 0.6, check_ori: bool = True) -> None:
        self.nnratio = nnratio
        self.check_orientation = check_ori

    def search_for_initialization(self, frame1: 'Frame', frame2: 'Frame', prev_matched: list['Geometry.Point2']) -> list[cv2.DMatch]:

        matches = self.match(frame1.descriptors, frame2.descriptors)
        # Return matches sorted by distance (https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)
        sorted_matches: list[cv2.DMatch] = sorted(matches, key=lambda x: x.distance)

        result = []
        for match in sorted_matches:
            if self.THRESH_LOW < match.distance < self.THRESH_HIGH:
                result.append(match)
        
        return result
    
    def map_points_by_descriptors(self, reference_frame: 'KeyFrame | Frame', other_frame: 'KeyFrame | Frame') -> dict[int, 'MapPoint']:
        matches = self.match(reference_frame.descriptors, other_frame.descriptors)

        result = {}
        map_points: dict[int: 'MapPoint'] = reference_frame.get_map_point_matches()
        for match in matches:
            if not match.queryIdx in map_points: continue
            map_point = map_points[match.queryIdx]
            result[match.trainIdx] = map_point
        return result

    @classmethod
    def match(clc, desc1: list[cv2.Mat], desc2: list[cv2.Mat]) -> list[cv2.DMatch]:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc1, desc2)
        return matches

    @classmethod
    def descriptor_distance(clc, desc1: cv2.Mat, desc2: cv2.Mat) -> int:
        match = clc.match(desc1.reshape(1, desc1.shape[0]), desc2.reshape(1, desc2.shape[0]))[0]
        return match.distance