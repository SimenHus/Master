import cv2
import numpy as np

from src.camera import Frame
from src.motion import Pose3
from src.measurements import CameraMeasurement


class KeyframeManager:
    def __init__(self, min_translation=0.1, min_rotation=5):
        self.keyframes: list[CameraMeasurement] = []

        self.previous_pose = None
        self.min_translation = min_translation
        self.min_rotation = min_rotation


    def determine_keyframe(self, measurement: CameraMeasurement) -> bool:
        meas_pose = measurement.latest_vessel_measurement.as_pose()
        if self.previous_pose is None:
            self.previous_pose = meas_pose
            return True

        delta_pose = meas_pose.between(self.previous_pose)
        translation = np.linalg.norm(delta_pose.translation())
        rotation = delta_pose.rotation().rpy()

        if translation >= self.min_translation or np.linalg.norm(rotation) >= np.radians(self.min_rotation):
            return True
        
        return False
    
    def add_keyframe(self, measurement: CameraMeasurement) -> None:
        self.previous_pose = measurement.latest_vessel_measurement.as_pose()
        self.keyframes.append(measurement)