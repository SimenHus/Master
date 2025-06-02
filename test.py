

from src.util import DataLoader, Geometry, Time
import numpy as np

BASE_FOLDER = './datasets/osl'
DATA_FOLDER = BASE_FOLDER + '/data'
stx_data = DataLoader.load_stx_data(DATA_FOLDER)
camera, Trc = DataLoader.load_stx_camera(BASE_FOLDER)


data0, data1 = stx_data[:2]


print(Geometry.SO3.RzRyRx(np.array([90, 0, 90])*np.pi / 180))