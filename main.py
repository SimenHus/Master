
from src import System
from src.structs import Camera
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


# def crude_data_loading_frames(path, camera_id) -> list[list[cv2.Mat, int]]:
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
        
#         frame = (image, vessel_measurement.timestep)

#         result.append(frame)

#     return result

def data_load_home(path, n):
    images = path + '/*.png'

    list_of_images = glob.glob(images)

    result = []
    n_files = len(list_of_images) if len(list_of_images) < n else n
    for i, image_file in enumerate(list_of_images[:n_files]):
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        timestep = int(image_file.split('/')[-1].strip('.png'))
        frame = (image, timestep)

        result.append(frame)

    return result

# https://arxiv.org/pdf/2007.11898
# See https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/Examples/Monocular/mono_euroc.cc
class CameraExtrinsicCalibration:

    def __init__(self) -> None:
        self.SLAM = System()


        # base_folder = '/mnt/c/Users/shustad/Desktop/Skole/Prosjektoppgave/data/osl'
        # data_folder = base_folder + '/2024-02-08/2024-02-08-14/Cam1/Lens0'
        # dataloader_file = base_folder + '/dataloader.json'

        data_folder = '/mnt/c/Users/simen/Desktop/Prog/Dataset/mav0/cam0/data'
        
        # with open(dataloader_file, 'r') as f: camera_dict = json.load(f)['Cam1']['Lens0']

        # self.camera = Camera.from_json(camera_id, camera_dict)
        n = 20
        self.images: list[list[cv2.Mat, int]] = data_load_home(data_folder, n)

    def start(self) -> None:
        
        for (image, timestep) in self.images:
            self.SLAM.track_monocular(image, timestep)

    def save_keyframes(self, filename: 'str') -> None:
        self.SLAM.save_keyframes(filename)

    def save_map_points(self, filename: 'str') -> None:
        self.SLAM.save_map_points(filename)

if __name__ == '__main__':
    app = CameraExtrinsicCalibration()
    app.start()

    OUTPUT_FOLDER = './output'
    app.save_keyframes(f'{OUTPUT_FOLDER}/KeyFrames')
    app.save_map_points(f'{OUTPUT_FOLDER}/MapPoints')
    
    exit()