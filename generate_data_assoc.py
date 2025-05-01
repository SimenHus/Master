import numpy as np
import cv2

import glob
import json

from queue import Queue
from time import sleep


import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from src.frontend import FrontendMain

from src.measurements import CameraMeasurement, VesselMeasurement
from src.camera import CameraModel, Frame

from src.visualization import FactorGraphVisualization, PlotVisualization


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
        camera_measurement = CameraMeasurement(vessel_measurement.timestep, 0, Frame(image), vessel_measurement)

        result.append(camera_measurement)

    return result

class Application:

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

        
        loaded_data = crude_data_loading(data_folder)
        self.camera_measurements: list[CameraMeasurement] = loaded_data

        self.running = True


    def start(self) -> None:
        self.frontend.start()

        for i, camera_meas in enumerate(self.camera_measurements):
            self.frontend_queue.put(camera_meas)

    def stop(self) -> None:
        self.running = False

        while not self.frontend_queue.empty(): sleep(0.1)
        self.frontend.stop()

        self.frontend.join()
        while not self.backend_queue.empty():
            print(self.backend_queue.get())


if __name__ == '__main__':
    app = Application()
    app.start()
    app.stop()


    exit()