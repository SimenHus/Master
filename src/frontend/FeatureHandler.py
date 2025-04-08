import cv2

from src.common import Frame

class FeatureHandler:
    orb = cv2.ORB.create()

    @classmethod
    def extract_features(clc, frame: Frame) -> tuple:
        keypoints, descriptors = clc.orb.detectAndCompute(frame.image, None)
        return keypoints, descriptors
    
    @staticmethod
    def match_features(frame1: Frame, frame2: Frame) -> None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(frame1.descriptors, frame2.descriptors)
        # Return matches sorted by distance (https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)
        return sorted(matches, key=lambda x: x.distance)