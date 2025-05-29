

from src.util import DataLoader, Geometry, Time
import numpy as np

BASE_FOLDER = './datasets/osl'
DATA_FOLDER = BASE_FOLDER + '/data'
stx_data = DataLoader.load_stx_data(DATA_FOLDER)
camera, Trc = DataLoader.load_stx_camera(BASE_FOLDER)


data0, data1 = stx_data[:2]


Twr0 = data0.state.pose
Twc0 = Twr0.compose(Trc)
Twcc = Geometry.SE3.NED_to_RDF(Twc0)

vec1 = Geometry.SE3.as_vector(Twr0)
vec2 = Geometry.SE3.as_vector(Twc0)
vec3 = Geometry.SE3.as_vector(Twcc)

print(f'Ref: {vec1}')
print(f'Cam NED: {vec2}')
print(f'Cam RDF: {vec3}')


# print(Geometry.SO3.RzRyRx(np.array([90, 0, 90])*np.pi/180))
# print(Geometry.SE3.NED_to_RDF_map())