
from src.structs import ImageData, STXData, Camera, LLA
from src.util import Time, Geometry, Geode
import cv2
import glob
import json
import numpy as np


def load_stx_images() -> list[ImageData]:
    img_path = data_path()
    suffix = 'jpg'
    images = img_path + f'/*.{suffix}'

    list_of_images = glob.glob(images)
    mask_path = config()['dir']['static_mask_path']
    if mask_path is None:
        mask = np.ones_like(cv2.imread(list_of_images[0], cv2.IMREAD_GRAYSCALE), dtype=np.uint8) * 255
    else:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    result = []
    n = config()['imgs_to_use']
    if n is None: n = len(list_of_images)
    n_files = len(list_of_images) if len(list_of_images) < n else n
    for i, image_file in enumerate(list_of_images[:n_files]):
        if mask_path is not None: pass
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        timestep = image_file.split('/')[-1].strip(f'.{suffix}')
        unix_timestep = Time.TimeConversion.STX_to_POSIX(timestep)
        frame = ImageData(image, unix_timestep, timestep, mask)
        result.append(frame)

    return sorted(result, key=lambda x: x.timestep)

def load_stx_data() -> list[STXData]:
    json_path = f'{data_path()}/*.json'
    list_of_jsons = glob.glob(json_path)
    n = config()['imgs_to_use']
    if n is None: n = len(list_of_jsons)
    n_files = len(list_of_jsons) if len(list_of_jsons) < n else n

    result = []
    ref_lla = None
    for i, json_file in enumerate(list_of_jsons[:n_files]):
        data_all = load_json(json_file)
        timestep = data_all['meas_time']
        data = data_all['own_vessel']
        unix_timestep = Time.TimeConversion.generic_to_POSIX(timestep)
        
        acc = np.array(data['acceleration'])
        atterror = np.array(data['atterror'])
        poserror = np.array(data['poserror'])
        attrate = np.array(data['attrate']) * np.pi / 180
        vel = np.array(data['velocity'])
        att = np.array(data['attitude']) * np.pi / 180
        lat, lon, alt = data['position']
        lla = LLA(lat, lon, alt)
        # if not ref_lla: ref_lla = lla
        pos = np.array([0, 0, 0])
        # pos = Geode.Transformation.LLA_to_NED(lla, ref_lla)
        state = Geometry.State(pos, vel, acc, poserror, att, attrate, atterror)

        result.append(STXData(state, lla, unix_timestep))
    sorted_list = sorted(result, key=lambda x: x.timestep)

    for i in range(len(sorted_list)):
        if not ref_lla: ref_lla = sorted_list[i].lla
        pos = Geode.Transformation.LLA_to_NED(sorted_list[i].lla, ref_lla)
        sorted_list[i].state.position = pos

    return sorted_list

def load_stx_camera(cam=1, lens=0) -> list[Camera, Geometry.SE3]:
    data = dataloader()
    lens_info = data[f'Cam{cam}'][f'Lens{lens}']
    K_list = lens_info['camera_matrix']
    pos = np.array(lens_info['location'])
    att = np.array(lens_info['rotation'])
    dist_coeffs = lens_info['distortion_coefficients']
    params = Camera.K_list_to_params(K_list)
    vals = np.append(att, pos)
    pose = Geometry.SE3.from_vector(vals, radians=False)
    return Camera(params, dist_coeffs), pose

def config() -> dict:
    return load_json('./config/Config.json')

def dataloader() -> dict:
    file_path = f'{config()["dir"]["dataloader"]}/dataloader.json'
    return load_json(file_path)

def data_path() -> str:
    return f'{config()["dir"]["data"]}'

def load_json(path) -> dict:
    with open(path, 'r') as f: return json.load(f)

def COLMAP_project_path() -> dict:
    return config()['COLMAP']['project_path']

def output_path() -> str:
    return config()['dir']['output']