
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
    THRESH_LOW: int = 5
    THRESH_HIGH: int = 50
    THRESH_SPATIAL = 0.1
    KNN_RATIO = 0.6

    def search_for_initialization(self, frame1: 'Frame', frame2: 'Frame') -> list[cv2.DMatch]:
        matches = self.match(frame1.descriptors, frame2.descriptors)
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
    def is_match(clc, mp1: 'MapPoint', mp2: 'MapPoint') -> bool:
        if clc.descriptor_distance(mp1.descriptor, mp2.descriptor) > clc.THRESH_HIGH: return False
        # if np.linalg.norm(mp1.get_world_pos() - mp2.get_world_pos()) > clc.THRESH_SPATIAL: return False
        return True
    

    @classmethod
    def filter_matches(clc, matches_knn, ratio) -> list[cv2.Mat]:
        # Lowes ratio filtering
        good_matches = []
        for (m, n) in matches_knn:
            if m.distance > clc.THRESH_HIGH or m.distance < clc.THRESH_LOW: continue
            if m.distance >= n.distance * ratio: continue
            good_matches.append(m)
        return good_matches
    

    @classmethod
    def match(clc, desc1: list[cv2.Mat], desc2: list[cv2.Mat]) -> list[cv2.DMatch]:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches_knn = matcher.knnMatch(desc1, desc2, k=2)
        if len(matches_knn) == 1: return matches_knn[0]
        matches = clc.filter_matches(matches_knn, clc.KNN_RATIO)
    
        return matches
    
    @classmethod
    def descriptor_distance(clc, desc1: cv2.Mat, desc2: cv2.Mat) -> int:
        return cv2.norm(desc1, desc2, cv2.NORM_HAMMING)