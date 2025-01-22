
import numpy as np
from gtsam import noiseModel
from gtsam import Pose2, Point2, Rot2
from gtsam import Pose3, Point3, Rot3

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


