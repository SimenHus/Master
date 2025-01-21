
import numpy as np
from gtsam import Pose2, noiseModel, Point2


class BaseClass:

    @staticmethod
    def RNG() -> np.random.Generator:
        return np.random.default_rng()

    @classmethod
    def default_noise_model(clc) -> noiseModel.Diagonal.Sigmas:
        return noiseModel.Diagonal.Sigmas(clc.std_dev_list())


class NoisyMeasurement2D(BaseClass):
    sigma_x = 0.1 # [m]
    sigma_y = 0.1 # [m]

    @classmethod
    def std_dev_list(clc) -> np.ndarray[2]:
        return [clc.sigma_x, clc.sigma_y]
    
    @classmethod
    def sample(clc, true_odometry: Pose2) -> Point2:
        noise_model = clc.default_noise_model()
        t = true_odometry.translation()
        cov = noise_model.covariance()
        noisy_t = clc.RNG().multivariate_normal(t, cov[:2, :2])
        
        return Point2(noisy_t[0], noisy_t[1])

class NoisyOdom2D(BaseClass):
    sigma_x = 0.5 # [m]
    sigma_y = 0.5 # [m]
    sigma_theta = 5.0 * np.pi/180 # [rad]

    @classmethod
    def std_dev_list(clc) -> np.ndarray[3]:
        return [clc.sigma_x, clc.sigma_y, clc.sigma_theta]
    
    @classmethod
    def sample(clc, true_odometry: Pose2) -> Pose2:
        noise_model = clc.default_noise_model()

        t = true_odometry.translation()
        r = true_odometry.rotation().theta()
        cov = noise_model.covariance()
        noisy_t = clc.RNG().multivariate_normal(t, cov[:2, :2])
        noisy_r = clc.RNG().normal(r, cov[2, 2])
        
        return Pose2(noisy_r, noisy_t)