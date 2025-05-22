
from src.structs import ImageData, PoseData, Camera
from src.util import Time, Geometry
import cv2
import glob
import json
import numpy as np



def load_stx_images(path, n = None) -> list[ImageData]:
    suffix = 'jpg'
    images = path + f'/*.{suffix}'

    list_of_images = glob.glob(images)

    result = []
    if n is None: n = len(list_of_images)
    n_files = len(list_of_images) if len(list_of_images) < n else n
    for i, image_file in enumerate(list_of_images[:n_files]):
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        timestep = image_file.split('/')[-1].strip(f'.{suffix}')
        unix_timestep = Time.TimeConversion.STX_to_POSIX(timestep)
        frame = ImageData(image, unix_timestep, timestep)
        result.append(frame)

    return sorted(result, key=lambda x: x.timestep)


def load_stx_poses(path, n = None) -> list[PoseData]:
    jsons = path + f'/*.json'
    list_of_jsons = glob.glob(jsons)

    result = []
    if n is None: n = len(list_of_jsons)
    n_files = len(list_of_jsons) if len(list_of_jsons) < n else n
    for i, json_file in enumerate(list_of_jsons[:n_files]):
        with open(json_file, 'r') as f: data_all = json.load(f)
        timestep = data_all['meas_time']
        data = data_all['own_vessel']
        unix_timestep = Time.TimeConversion.generic_to_POSIX(timestep)
        
        pos = np.array(data['position'])
        att = np.array(data['attitude'])
        vals = np.append(att, pos)
        pose = Geometry.SE3.from_STX(vals)
        result.append(PoseData(pose, unix_timestep))

    return sorted(result, key=lambda x: x.timestep)


def load_stx_camera(path, cam=1, lens=0) -> list[Camera, Geometry.SE3]:
    dataloader = path + '/dataloader.json'
    with open(dataloader, 'r') as f: data = json.load(f)
    lens_info = data[f'Cam{cam}'][f'Lens{lens}']
    K_list = lens_info['camera_matrix']
    pos = np.array(lens_info['location'])
    att = np.array(lens_info['rotation'])
    dist_coeffs = lens_info['distortion_coefficients']
    params = Camera.K_list_to_params(K_list)
    vals = np.append(att, pos)
    pose = Geometry.SE3.from_STX(vals)
    return Camera(params, dist_coeffs), pose