

from gtsam import Pose2, noiseModel
import numpy as np


def apply_noise_pose2(pose: Pose2, noise: noiseModel) -> Pose2:
    RNG = np.random.default_rng()
    t = pose.translation()
    r = pose.rotation().theta()
    cov = noise.covariance()
    noisy_t = RNG.multivariate_normal(t, cov[:2, :2])
    noisy_r = RNG.normal(r, cov[2, 2])
    
    return Pose2(noisy_r, noisy_t)