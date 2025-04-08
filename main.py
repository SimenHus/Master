
import gtsam
from gtsam import ISAM2Params, ISAM2, Marginals
from gtsam import NonlinearFactorGraph, Values, Symbol
from gtsam import PriorFactorPose3, BetweenFactorPose3, BearingRangeFactor3D, PriorFactorPoint3, Pose3, Point3, Rot3, GPSFactor
from gtsam import noiseModel, LevenbergMarquardtOptimizer, DoglegOptimizer
from gtsam.symbol_shorthand import X, L, T

import numpy as np
import cv2

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from src.backend import SLAM
from src.frontend import FeatureExtractor
from src.common import Frame

from src.visualization.GraphVisualization import FactorGraphVisualization
from src.visualization.PlotVisualization import plot_graph3D


class CameraExtrinsicEstimation:

    def __init__(self) -> None:
        self.SLAM = SLAM()

        image = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)
        frame = Frame(0, image)
        frame = FeatureExtractor.extract_features(frame)
        img2 = cv2.drawKeypoints(frame.image, frame.keypoints, None, color=(0,255,0), flags=0)
        plt.imshow(img2)
        plt.show()

    def run(self) -> None:
        return


if __name__ == '__main__':
    app = CameraExtrinsicEstimation()
    app.run()

    exit()