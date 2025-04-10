
import numpy as np
import cv2

import glob
import csv

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from src.backend import SLAM
from src.frontend import FeatureHandler, KeyframeManager
from src.common import Frame
from src.motion import Pose3
from src.visualization import draw_matches

from src.visualization.GraphVisualization import FactorGraphVisualization
from src.visualization.PlotVisualization import plot_graph3D


class CameraExtrinsicEstimation:

    def __init__(self) -> None:
        self.SLAM = SLAM()
        self.kf_manager = KeyframeManager(0, 0)

        image1 = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)
        frame1 = Frame(0, image1)

        image2 = cv2.imread('test3.png', cv2.IMREAD_GRAYSCALE)
        frame2 = Frame(1, image2)

        # image_paths = glob.glob('C:\Users\simen\Desktop\Prog\Dataset\mav0\cam0\data')
        image_csv = '/mnt/c/Users/simen/Desktop/Prog/Dataset/mav0/cam0/data.csv'

        images = {}
        
        with open(image_csv) as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i == 0: continue
                timestamp, filename = int(row[0]), row[1]
                images[timestamp] = filename


        self.images = [frame1, frame2]

    def run(self) -> None:

        for image in self.images:
            keypoints, descriptors = FeatureHandler.extract_features(image)
            image.set_features(keypoints, descriptors)
            pose = Pose3()
            is_keyframe = self.kf_manager.determine_keyframe(pose)
            if is_keyframe: self.kf_manager.add_keyframe(image)
        




if __name__ == '__main__':
    app = CameraExtrinsicEstimation()
    app.run()

    frame1,frame2 = app.kf_manager.keyframes
    matched_image = draw_matches(frame1, frame2)

    plt.imshow(matched_image)
    plt.show()


    # img2 = cv2.drawKeypoints(frame.image, frame.keypoints, None, color=(0,255,0), flags=0)
    # plt.imshow(img2)
    # plt.show()

    exit()