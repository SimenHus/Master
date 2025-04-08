
import cv2
import numpy as np

from src.motion import Pose3

class LoopClosure:
    pass
    # def detect_loop_closure(current_pose, pose_history, threshold=1.0):
    #     current_pos = np.array(current_pose.translation())
    #     for idx, past_pose in enumerate(pose_history):
    #         dist = np.linalg.norm(np.array(past_pose.translation()) - current_pos)
    #         if dist < threshold:
    #             return idx
    #     return None