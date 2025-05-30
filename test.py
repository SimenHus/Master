

from src.util import DataLoader, Geometry, Time
import numpy as np

BASE_FOLDER = './datasets/osl'
DATA_FOLDER = BASE_FOLDER + '/data'
stx_data = DataLoader.load_stx_data(DATA_FOLDER)
camera, Trc = DataLoader.load_stx_camera(BASE_FOLDER)


data0, data1 = stx_data[:2]


Twr0 = data0.state.pose
Twr1 = data1.state.pose
dt = Time.TimeConversion.dt_POSIX_to_SECONDS(data1.timestep - data0.timestep)
twist0 = data0.state.twist

Twr1_guess = Geometry.SE3.Expmap(twist0).compose(Twr0)
Twr1_guess = Twr0.compose(Geometry.SE3.Expmap(twist0))

vec1 = Geometry.SE3.as_vector(Twr0)
vec2 = Geometry.SE3.as_vector(Twr1)
vec3 = Geometry.SE3.as_vector(Twr1_guess)

print(f'Twr0: {vec1}')
print(f'Twr1 GT: {vec2}')
print(f'Trw1 Guess: {vec3}')
print(f'Error: {vec3 - vec2}')