
import numpy as np
import cv2

import glob
import csv

import json

from queue import Queue
from time import sleep


import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from src.backend import BackendMain
from src.frontend import FrontendMain

from src.common import Frame
from src.measurements import CameraMeasurement, VesselMeasurement
from src.models import CameraModel

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
        camera_measurement = CameraMeasurement(vessel_measurement.timestamp, 0, Frame(image), vessel_measurement)

        result.append(camera_measurement)

    return result



class CameraExtrinsicEstimation:

    def __init__(self) -> None:
        base_folder = '/mnt/c/Users/shustad/Desktop/Skole/Prosjektoppgave/data/osl'
        data_folder = base_folder + '/2024-02-08/2024-02-08-14/Cam1/Lens0'

        dataloader_file = base_folder + '/dataloader.json'
        
        with open(dataloader_file, 'r') as f: camera_dict = json.load(f)['Cam1']['Lens0']

        self.frontend_queue = Queue()
        self.backend_queue = Queue()

        camera = CameraModel.from_json(camera_dict)
        self.cameras = [camera]

        self.images: list[Frame] = []

        self.frontend = FrontendMain(self.frontend_queue, self.backend_queue, self.cameras)
        self.backend = BackendMain(self.backend_queue)

        
        loaded_data = crude_data_loading(data_folder)
        self.camera_measurements: list[CameraMeasurement] = loaded_data

        self.running = True


    def start(self) -> None:
        self.frontend.start()
        self.backend.start()

        for i, camera_meas in enumerate(self.camera_measurements):
            timestep = i + 1
            camera_id = 0

            self.frontend_queue.put(camera_meas)

    def stop(self) -> None:
        self.running = False

        while not self.frontend_queue.empty() or not self.backend_queue.empty(): sleep(0.1)
        self.frontend.stop()
        self.backend.stop()

        self.frontend.join()
        self.backend.join()


if __name__ == '__main__':
    app = CameraExtrinsicEstimation()
    app.start()
    app.stop()

    output_folder = './'
    FactorGraphVisualization.draw_factor_graph(output_folder, app.backend.SLAM.graph, app.backend.SLAM.current_estimate)
    print(app.backend.SLAM.current_estimate)

    # print(app.frontend.)

    exit()