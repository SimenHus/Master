import cv2

from src.common import Frame

class FeatureExtractor:
    orb = cv2.ORB.create()

    @classmethod
    def extract_features(clc, frame: Frame) -> Frame:
        keypoints, descriptors = clc.orb.detectAndCompute(frame.image, None)
        frame.keypoints = keypoints
        frame.descriptors = descriptors

        return frame
