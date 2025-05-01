
from src.camera import Feature, Frame
import cv2
import numpy as np

from .PlotVisualization import PlotVisualization
from .GraphVisualization import FactorGraphVisualization

def draw_matches(frame1: Frame, frame2: Frame) -> np.ndarray:
    matches = Feature.match_features(frame1, frame2)
    image = cv2.drawMatches(
        frame1.image, frame1.keypoints,
        frame2.image, frame2.keypoints,
        matches[:10], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return image