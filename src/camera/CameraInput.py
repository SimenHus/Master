
import cv2

from dataclasses import dataclass

import numpy as np

class Descriptor(np.ndarray): # Abstraction of descriptor
    pass


class Keypoint(cv2.KeyPoint): # Abstraction of Keypoint
    pass

class Match(cv2.DMatch):
    pass


@dataclass
class Feature:
    keypoint: Keypoint
    descriptor: Descriptor

    @classmethod
    def match_features(clc, frame1: 'Frame', frame2: 'Frame') -> list[Match]:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(frame1.descriptors, frame2.descriptors)
        # Return matches sorted by distance (https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)
        return sorted(matches, key=lambda x: x.distance)
    
@dataclass
class Frame:
    image: np.ndarray
    id: int = -1
    features: list[Feature] | None = None
    orb = cv2.ORB.create()

    def set_features(self, features: list[Feature]) -> None:
        self.features = features

    def extract_features(self) -> list[Feature]:
        keypoints, descriptors = self.orb.detectAndCompute(self.image, None)
        features = [Feature(kp, desc) for (kp, desc) in zip(keypoints, descriptors)]
        self.set_features(features)
        return features

    @property
    def keypoints(self) -> list[Keypoint]:
        return [feature.keypoint for feature in self.features]
    
    @property
    def descriptors(self) -> list[Descriptor]:
        return [feature.descriptor for feature in self.features]