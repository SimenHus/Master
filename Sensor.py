
import numpy as np
from gtsam import noiseModel
from gtsam import Pose2, Point2, Rot2
from gtsam import Pose3, Point3, Rot3


class SimulatedSensorBaseClass:
    RNG = np.random.default_rng()

    @classmethod
    def default_noise_model(clc) -> noiseModel.Diagonal.Sigmas:
        return noiseModel.Diagonal.Sigmas(clc.std_dev_list)


class NoisyMeasurement2D(SimulatedSensorBaseClass):
    sigma_x = 0.01 # [m]
    sigma_y = 0.01 # [m]
    std_dev_list = [sigma_x, sigma_y]
    
    @classmethod
    def sample(clc, true_odometry: Pose2) -> Point2:
        noise_model = clc.default_noise_model()
        t = true_odometry.translation()
        cov = noise_model.covariance()
        noisy_t = clc.RNG.multivariate_normal(t, cov[:2, :2])
        
        return Point2(*noisy_t)

class NoisyOdom2D(SimulatedSensorBaseClass):
    sigma_x = 0.5 # [m]
    sigma_y = 0.5 # [m]
    sigma_theta = 5.0 * np.pi/180 # [rad]
    std_dev_list = [sigma_x, sigma_y, sigma_theta]

    
    @classmethod
    def sample(clc, true_odometry: Pose2) -> Pose2:
        noise_model = clc.default_noise_model()

        t = true_odometry.translation()
        r = true_odometry.rotation().theta()
        cov = noise_model.covariance()
        noisy_t = clc.RNG.multivariate_normal(t, cov[:2, :2])
        noisy_r = clc.RNG.normal(r, cov[2, 2])
        
        return Pose2(noisy_r, noisy_t)
    

class NoisyOdom3D(SimulatedSensorBaseClass):
    sigma_x = 0.005 # [m]
    sigma_y = 0.005 # [m]
    sigma_z = 0.005 # [m]
    sigma_theta = 5.0 * np.pi/180 # [rad]
    sigma_phi = 5.0 * np.pi/180 # [rad]
    sigma_psi = 5.0 * np.pi/180 # [rad]
    std_dev_list = [sigma_x, sigma_y, sigma_z, sigma_theta, sigma_phi, sigma_psi]

    @classmethod
    def sample(clc, true_odometry: Pose3) -> Pose3:
        noise_model = clc.default_noise_model()

        cov = noise_model.covariance()

        translation_noise = clc.RNG.multivariate_normal(np.zeros(3), cov[:3, :3])
        rotation_noise = clc.RNG.multivariate_normal(np.zeros(3), cov[3:, 3:])

        noisy_translation = Point3(*translation_noise)
        noisy_rotation = Rot3.Expmap(rotation_noise)
        
        group_noise = Pose3(noisy_rotation, noisy_translation)

        return true_odometry * group_noise
    


class NoisyMeasurement3D(SimulatedSensorBaseClass):
    sigma_x = 0.1 # [m]
    sigma_y = 0.1 # [m]
    sigma_z = 0.1 # [m]
    std_dev_list = [sigma_x, sigma_y, sigma_z]
    
    @classmethod
    def sample(clc, true_odometry: Pose3) -> Point3:
        noise_model = clc.default_noise_model()
        t = true_odometry.translation()
        cov = noise_model.covariance()
        noisy_t = clc.RNG.multivariate_normal(t, cov[:3, :3])

        return Point3(*noisy_t)