
import numpy as np
from gtsam import noiseModel
from gtsam import Pose2, Point2, Rot2
from gtsam import Pose3, Point3, Rot3

from .RigidMotion import GroupIdentifier
from .NoiseModel import NoiseModel


class SimulatedMeasurement:
    def sample(true_state: Pose3 | Pose2, noise_model: NoiseModel):
        group = GroupIdentifier.identify(true_state)
        cov = noise_model.default_noise_model().covariance()

        t_dim = len(true_state.translation())
        r_dim = cov.shape[0] - t_dim

        translation_noise = np.zeros(t_dim)
        rotation_noise = np.zeros(t_dim)

        if t_dim > 0: translation_noise = noise_model.RNG.multivariate_normal(np.zeros(t_dim), cov[:t_dim, :t_dim])
        if r_dim > 0: rotation_noise = noise_model.RNG.multivariate_normal(np.zeros(r_dim), cov[t_dim:, t_dim:])


        noisy_translation = group.point(*translation_noise)
        noisy_rotation = group.rot.Expmap(rotation_noise)
        
        group_noise = group.pose(noisy_rotation, noisy_translation)

        return true_state * group_noise