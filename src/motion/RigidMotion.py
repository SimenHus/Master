

import gtsam
from gtsam import Point3
import numpy as np

class Pose3(gtsam.Pose3): # Abstraction of the gtsam Pose3 class
    pass

class Rot3(gtsam.Rot3): # Abstraction of gtsam Rot3 class
    pass

# class Point3(gtsam.Point3): # Abstraction of gtsam Point3 class
#     pass



class Pose3Noise: # Abstraction of noise models in gtsam
    
    def __init__(self, pos_noise: np.ndarray[3], rot_noise: np.ndarray[3]) -> None:
        self.att_noise = rot_noise
        self.pos_noise = pos_noise

    def noise_model(self) -> gtsam.noiseModel:
        noise = np.append(self.pos_noise, self.att_noise)
        return gtsam.noiseModel.Diagonal.Sigmas(noise)
    