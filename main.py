
from src import System
from src.structs import Camera, CameraMeasurement, VesselMeasurement
import glob
import json
import cv2

# def crude_data_loading(path) -> list[CameraMeasurement]: # For KCC
#     images = path + '/*.jpg'
#     json_files = path + '/*.json'

#     list_of_images = glob.glob(images)
#     list_of_jsons = glob.glob(json_files)

#     result = []
#     n = 4
#     for i, (image_file, json_file) in enumerate(zip(list_of_images[:n], list_of_jsons[:n])):
#         image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
#         with open(json_file, 'r') as f:
#             json_data = json.load(f)

#         vessel_measurement = VesselMeasurement.from_json(json_data)
#         camera_measurement = CameraMeasurement(vessel_measurement.timestep, 0, Frame(image), vessel_measurement)

#         result.append(camera_measurement)

#     return result


def crude_data_loading_frames(path, camera_id) -> list[list[cv2.Mat, int]]:
    images = path + '/*.jpg'
    json_files = path + '/*.json'

    list_of_images = glob.glob(images)
    list_of_jsons = glob.glob(json_files)

    result = []
    n = 4
    for i, (image_file, json_file) in enumerate(zip(list_of_images[:n], list_of_jsons[:n])):
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        with open(json_file, 'r') as f:
            json_data = json.load(f)

        vessel_measurement = VesselMeasurement.from_json(json_data)
        
        frame = (image, vessel_measurement.timestep)

        result.append(frame)

    return result

# https://arxiv.org/pdf/2007.11898
# See https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/Examples/Monocular/mono_euroc.cc
class CameraExtrinsicCalibration:

    def __init__(self) -> None:
        self.SLAM = System()


        base_folder = '/mnt/c/Users/shustad/Desktop/Skole/Prosjektoppgave/data/osl'
        data_folder = base_folder + '/2024-02-08/2024-02-08-14/Cam1/Lens0'
        dataloader_file = base_folder + '/dataloader.json'
        
        with open(dataloader_file, 'r') as f: camera_dict = json.load(f)['Cam1']['Lens0']

        camera_id = 1
        self.camera = Camera.from_json(camera_id, camera_dict)

        self.images: list[list[cv2.Mat, int]] = crude_data_loading_frames(data_folder, camera_id)

    def start(self) -> None:
        
        for (image, timestep) in self.images:
            self.SLAM.track_monocular(image, timestep)



if __name__ == '__main__':
    app = CameraExtrinsicCalibration()
    app.start()
    
    exit()