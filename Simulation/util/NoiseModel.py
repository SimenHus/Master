
import numpy as np
from gtsam import noiseModel

class NoiseModel:
    RNG = np.random.default_rng()

    @classmethod
    def default_noise_model(clc) -> noiseModel.Diagonal.Sigmas:
        return noiseModel.Diagonal.Sigmas(clc.std_dev_list)
    

class Position3D(NoiseModel):
    sigma_x = 0.1 # [m]
    sigma_y = 0.1 # [m]
    sigma_z = 0.1 # [m]
    std_dev_list = [sigma_x, sigma_y, sigma_z]

class Position2D(NoiseModel):
    sigma_x = 0.1 # [m]
    sigma_y = 0.1 # [m]
    std_dev_list = [sigma_x, sigma_y]

class Odometry3D(NoiseModel):
    sigma_x = 0.005 # [m]
    sigma_y = 0.005 # [m]
    sigma_z = 0.005 # [m]
    sigma_theta = 5.0 * np.pi/180 # [rad]
    sigma_phi = 5.0 * np.pi/180 # [rad]
    sigma_psi = 5.0 * np.pi/180 # [rad]
    std_dev_list = [sigma_x, sigma_y, sigma_z, sigma_theta, sigma_phi, sigma_psi]


class Odometry2D(NoiseModel):
    sigma_x = 0.5 # [m]
    sigma_y = 0.5 # [m]
    sigma_theta = 5.0 * np.pi/180 # [rad]
    std_dev_list = [sigma_x, sigma_y, sigma_theta]


class Landmark3D(NoiseModel):
    sigma_x = 0.1 # [m]
    sigma_y = 0.1 # [m]
    sigma_z = 0.1 # [m]
    std_dev_list = [sigma_x, sigma_y, sigma_z]

class Special(NoiseModel):
    sigma_x = 0.1 # [m]
    sigma_y = 0.1 # [m]
    sigma_z = 0.1 # [m]
    std_dev_list = [sigma_x, sigma_y, sigma_z]