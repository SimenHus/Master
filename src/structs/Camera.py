
import gtsam
import cv2
from dataclasses import dataclass
import numpy as np

from src.util import Pose3



class Descriptor(np.ndarray): # Abstraction of descriptor
    def __hash__(self) -> int:
        return hash(self)


class KeyPoint(cv2.KeyPoint): # Abstraction of Keypoint
    pass

class Match(cv2.DMatch):
    pass


@dataclass
class Feature:
    keypoint: KeyPoint
    descriptor: Descriptor

    @classmethod
    def match_features(clc, frame1: 'Frame', frame2: 'Frame') -> list[Match]:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(frame1.descriptors, frame2.descriptors)
        # Return matches sorted by distance (https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)
        return sorted(matches, key=lambda x: x.distance)
    

@dataclass
class Frame:
    image: cv2.MatLike
    camera_id: int
    timestep: int
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
    def keypoints(self) -> list[KeyPoint]:
        return [feature.keypoint for feature in self.features]
    
    @property
    def descriptors(self) -> list[Descriptor]:
        return [feature.descriptor for feature in self.features]


class CameraNoiseModel:
    def cov(self) -> gtsam.noiseModel:
        return gtsam.noiseModel.Diagonal.Sigmas([1, 1])




@dataclass
class KeyFrame:
    frame: Frame
    camera_pose: Pose3

    @property
    def camera_id(self) -> int:
        return self.frame.camera_id
    
    @property
    def timestep(self) -> int:
        return self.frame.timestep

    def __eq__(self, other) -> bool:
        if not isinstance(other, KeyFrame): return False
        return self.timestep == other.timestep and self.camera_id == other.camera_id

    def __hash__(self) -> int:
        return hash(tuple(self.timestep, self.camera_id))


@dataclass
class CameraModel:
    dim: tuple[int, int, int] # Camera dimensions in pixels (height, width, channels)
    focal_length: tuple[int, int] # Focal length in pixels
    principal_point: tuple[int, int] # Optical center in pixels
    distortion_coefficients: tuple # Distortion coefficients
    skew: float = 0.

    @property
    def K(self) -> np.ndarray[3, 3]:
        return np.array([
            [self.focal_length[0], self.skew, self.principal_point[0]],
            [0, self.focal_length[1], self.principal_point[1]],
            [0, 0, 1]
        ])
    
    @property
    def noise_model(self) -> CameraNoiseModel:
        return CameraNoiseModel()
    
    @staticmethod
    def from_json(json_dict: dict) -> 'CameraModel':
        K = np.array(json_dict['camera_matrix']).reshape((3, 3))
        return CameraModel(
            dim = (2056, 2464, 3),
            focal_length = (K[0, 0], K[1, 1]),
            principal_point = (K[0, 2], K[1, 2]),
            distortion_coefficients = json_dict['distortion_coefficients'],
            skew = K[0, 1]
        )


@dataclass
class Camera:
    id: int
    model: CameraModel

    @property
    def K(self) -> np.ndarray[3, 3]:
        return self.model.K
    
    @property
    def noise_model(self) -> CameraNoiseModel:
        return self.model.noise_model
    

    @staticmethod
    def from_json(camera_id: int, json_dict: dict) -> 'Camera':
        model = CameraModel.from_json(json_dict)
        return Camera(camera_id, model)
    