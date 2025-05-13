
import cv2
import numpy as np


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
    THRESH_LOW: int = 15
    THRESH_HIGH: int = 50
    HISTO_LENGTH: int = 30

    def __init__(self, check_ori: bool = True) -> None:
        self.check_orientation = check_ori

    def search_for_initialization(self, frame1: 'Frame', frame2: 'Frame') -> list[cv2.DMatch]:

        matches = self.match(frame1.descriptors, frame2.descriptors)
        matches = self.filter_matches(frame1.keypoints_und, frame2.keypoints_und, matches)
        # Return matches sorted by distance (https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)
        return matches
    
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
    def remove_outliers(clc, kp1: list[cv2.KeyPoint], kp2: list[cv2.KeyPoint], matches: list[cv2.DMatch]) -> list[cv2.DMatch]:
        outlier_matches = clc.detect_outliers(kp1, kp2, matches)
        result = []
        for match in matches:
            if match not in outlier_matches:
                result.append(match)
        return result
    
    @classmethod
    def detect_outliers(clc, kp1: list[cv2.KeyPoint], kp2: list[cv2.KeyPoint], matches: list[cv2.DMatch]) -> list[cv2.DMatch]:
        # Get the matching keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Estimate homography or affine transformation and mask out outliers
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Filter matches based on inlier mask
        # inlier_matches = [m for i, m in enumerate(matches) if mask[i]]
        outlier_matches = [m for i, m in enumerate(matches) if not mask[i]]

        return outlier_matches
    
            
    @classmethod
    def is_match(clc, desc1: cv2.Mat, desc2: cv2.Mat) -> bool:
        return clc.descriptor_distance(desc1, desc2) < clc.THRESH_LOW * 2
    

    @classmethod
    def filter_matches(clc, kp1: list[cv2.KeyPoint], kp2: list[cv2.KeyPoint], matches: list[cv2.DMatch]) -> list[cv2.Mat]:
        # sorted_matches: list[cv2.DMatch] = sorted(matches, key=lambda x: x.distance) # Sort matches by distance
        matches = clc.remove_outliers(kp1, kp2, matches)
        result = []
        for match in matches:
            if clc.THRESH_LOW < match.distance < clc.THRESH_HIGH:
                result.append(match)
        return result
    

    @classmethod
    def match(clc, desc1: list[cv2.Mat], desc2: list[cv2.Mat]) -> list[cv2.DMatch]:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(desc1, desc2)
    
        return matches
    
    @classmethod
    def descriptor_distance(clc, desc1: cv2.Mat, desc2: cv2.Mat) -> int:
        match = clc.match(desc1.reshape(1, desc1.shape[0]), desc2.reshape(1, desc2.shape[0]))[0]
        return match.distance