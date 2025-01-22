
import numpy as np
from gtsam import noiseModel
from gtsam import Pose2, Point2, Rot2
from gtsam import Pose3, Point3, Rot3

from .Generators import *
from .NoiseModel import NoiseModel


class SimulatedMeasurement:
    def sample(true_state: Pose3 | Pose2, generator: Generator3D | Generator2D, noise_model: NoiseModel):
        cov = noise_model.default_noise_model().covariance()

        t_dim = len(true_state.translation())
        r_dim = 1 if type(true_state) == Pose2 else 3

        translation_noise = noise_model.RNG.multivariate_normal(np.zeros(t_dim), cov[:t_dim, :t_dim])
        try:
            rotation_noise = noise_model.RNG.multivariate_normal(np.zeros(r_dim), cov[t_dim:, t_dim:])
        except:
            rotation_noise = np.zeros(r_dim)

        noisy_translation = generator.point(*translation_noise)
        noisy_rotation = generator.rot.Expmap(rotation_noise)
        
        group_noise = generator.pose(noisy_rotation, noisy_translation)

        return true_state * group_noise

class SimulatedOdometry:
    sigma_x = 0.005 # [m]
    sigma_y = 0.005 # [m]
    sigma_z = 0.005 # [m]
    sigma_theta = 5.0 * np.pi/180 # [rad]
    sigma_phi = 5.0 * np.pi/180 # [rad]
    sigma_psi = 5.0 * np.pi/180 # [rad]

    RNG = np.random.default_rng()
    std_dev_2D = [sigma_x, sigma_y, sigma_theta]
    std_dev_3D = [sigma_x, sigma_y, sigma_z, sigma_theta, sigma_phi, sigma_psi]

    @classmethod
    def default_noise_model_2D(clc) -> noiseModel.Diagonal.Sigmas:
        return noiseModel.Diagonal.Sigmas(clc.std_dev_2D)
    
    @classmethod
    def default_noise_model_3D(clc) -> noiseModel.Diagonal.Sigmas:
        return noiseModel.Diagonal.Sigmas(clc.std_dev_3D)
    
    @classmethod
    def sample_2D(clc, true_odometry: Pose2) -> Pose2:
        return clc._sample(true_odometry, Generator2D, clc.default_noise_model_2D())
    
    @classmethod
    def sample_3D(clc, true_odometry: Pose3) -> Pose3:
        return clc._sample(true_odometry, Generator3D, clc.default_noise_model_3D())
    
    @classmethod
    def _sample(clc, true_odometry: Pose3 | Pose2, generator: Generator3D | Generator2D, noise_model: noiseModel.Diagonal.Sigmas) -> Pose3 | Pose2:

        cov = noise_model.covariance()

        t_dim = len(true_odometry.translation())
        r_dim = 1 if type(true_odometry) == Pose2 else 3

        translation_noise = clc.RNG.multivariate_normal(np.zeros(t_dim), cov[:t_dim, :t_dim])
        rotation_noise = clc.RNG.multivariate_normal(np.zeros(r_dim), cov[t_dim:, t_dim:])

        noisy_translation = generator.point(*translation_noise)
        noisy_rotation = generator.rot.Expmap(rotation_noise)
        
        group_noise = generator.pose(noisy_rotation, noisy_translation)

        return true_odometry * group_noise



class SimulatedPositionMeasurement:
    RNG = np.random.default_rng()
    sigma_x = 0.1 # [m]
    sigma_y = 0.1 # [m]
    sigma_z = 0.1 # [m]
    std_dev_list = [sigma_x, sigma_y, sigma_z]

    @classmethod
    def default_noise_model(clc) -> noiseModel.Diagonal.Sigmas:
        return noiseModel.Diagonal.Sigmas(clc.std_dev_list)
    
    @classmethod
    def sample_2D(clc, true_odometry: Pose2) -> Pose2:
        return clc._sample(true_odometry, Generator2D)
    
    @classmethod
    def sample_3D(clc, true_odometry: Pose3) -> Pose3:
        return clc._sample(true_odometry, Generator3D)

    @classmethod
    def _sample(clc, true_odometry: Pose3 | Pose2, generator: Generator3D | Generator3D) -> Pose3 | Pose2:
        noise_model = clc.default_noise_model()
        t = true_odometry.translation()
        dim = len(t)
        cov = noise_model.covariance()
        noisy_t = clc.RNG.multivariate_normal(t, cov[:dim, :dim])

        return generator.pose(generator.rot(), generator.point(noisy_t))