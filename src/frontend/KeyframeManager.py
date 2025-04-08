import cv2
import numpy as np

from src.common import Frame
from src.motion import Pose3


class KeyframeManager:
    def __init__(self, min_translation=0.1, min_rotation=5):
        self.keyframes: list[Frame] = []

        self.previous_pose = None
        self.min_translation = min_translation
        self.min_rotation = min_rotation


    def determine_keyframe(self, current_pose: Pose3) -> bool:

        if self.previous_pose is None:
            self.previous_pose = current_pose
            return True

        delta_pose = current_pose.between(self.previous_pose)
        translation = np.linalg.norm(delta_pose.translation())
        rotation = delta_pose.rotation().rpy()

        if translation >= self.min_translation or np.linalg.norm(rotation) >= np.radians(self.min_rotation):
            self.previous_pose = current_pose
            return True
        
        return False
    
    def add_keyframe(self, frame: Frame) -> None:
        self.keyframes.append(frame)