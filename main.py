
import numpy as np
import cv2

import glob
import csv

import json

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from src.backend import SLAM
from src.frontend import FeatureHandler, KeyframeManager
from src.common import Frame, CameraMeasurement, VesselMeasurement
from src.motion import CameraNoise
from src.visualization import draw_matches

from src.visualization.GraphVisualization import FactorGraphVisualization
from src.visualization.PlotVisualization import plot_graph3D


def crude_data_loading(path) -> list[CameraMeasurement]: # For KCC
    images = path + '/*.jpg'
    json_files = path + '/*.json'

    list_of_images = glob.glob(images)
    list_of_jsons = glob.glob(json_files)

    result = []
    n = 10
    for i, (image_file, json_file) in enumerate(zip(list_of_images[:n], list_of_jsons[:n])):
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        with open(json_file, 'r') as f:
            json_data = json.load(f)

        vessel_measurement = VesselMeasurement.from_json(json_data)
        camera_measurement = CameraMeasurement(image, vessel_measurement, vessel_measurement.timestamp)

        result.append(camera_measurement)

    return result



class CameraExtrinsicEstimation:

    def __init__(self) -> None:
        self.SLAM = SLAM()
        self.kf_manager = KeyframeManager(0, 0)
        self.images: list[Frame] = []

        # data_folder = '/mnt/c/Users/simen/Desktop/Prog/Dataset/mav0/cam0/data'
        data_folder = '/mnt/c/Users/shustad/Desktop/Skole/Prosjektoppgave/data/osl/2024-02-08/2024-02-08-14/Cam1/Lens0'
        # image_paths = glob.glob('C:\Users\simen\Desktop\Prog\Dataset\mav0\cam0\data')
        # image_csv = '/mnt/c/Users/simen/Desktop/Prog/Dataset/mav0/cam0/data.csv'

        loaded_data = crude_data_loading(data_folder)
        
        self.camera_measurements: list[CameraMeasurement] = loaded_data

        self.K = np.array([
            [2407.240, 0, 1245.608],
            [0, 2404.539, 1004.407],
            [0, 0, 1]
        ])


    def run(self) -> None:

        # for image in self.images:
        #     keypoints, descriptors = FeatureHandler.extract_features(image)
        #     image.set_features(keypoints, descriptors)
        #     pose = Pose3()
        #     is_keyframe = self.kf_manager.determine_keyframe(pose)
        #     if is_keyframe:
        #         self.kf_manager.add_keyframe(image)

        for i, camera_meas in enumerate(self.camera_measurements):
            id = i + 1
            frame = Frame(id, camera_meas.image)
            keypoints, descriptors = FeatureHandler.extract_features(frame)
            frame.set_features(keypoints, descriptors)

            pose = camera_meas.latest_vessel_measurement.as_pose()
            pose_noise = camera_meas.latest_vessel_measurement.pose_noise()
            is_keyframe = self.kf_manager.determine_keyframe(pose)
            if is_keyframe or True:
                self.kf_manager.add_keyframe(frame)
                self.SLAM.pose_measurement(id, pose, pose_noise)
                landmark_pixels = frame.keypoints[0].pt
                pixel_noise = CameraNoise(np.array([1, 1]))
                predicted_pos = [100, 0, 0]
                self.SLAM.landmark_measurement(id, id, predicted_pos, self.K, landmark_pixels, pixel_noise)
                if i > 0:
                    odom = pose.between(self.camera_measurements[i-1].latest_vessel_measurement.as_pose())
                    odom_noise = pose_noise
                    self.SLAM.odometry_measurement(id - 1, id, odom, odom_noise)
                    self.SLAM.optimize()
        

if __name__ == '__main__':
    app = CameraExtrinsicEstimation()
    app.run()

    frame1,frame2 = app.kf_manager.keyframes[:2]
    matched_image = draw_matches(frame1, frame2)

    output_folder = './'
    FactorGraphVisualization.draw_factor_graph(output_folder, app.SLAM.graph, app.SLAM.current_estimate())
    print(app.SLAM.current_estimate())

    # plt.imshow(matched_image)
    # plt.show()

    # img2 = cv2.drawKeypoints(frame.image, frame.keypoints, None, color=(0,255,0), flags=0)
    # plt.imshow(img2)
    # plt.show()

    exit()