import gtsam
import numpy as np

class Pose3Noise: # Abstraction of noise models in gtsam
    
    def __init__(self, pos_noise: np.ndarray[3], rot_noise: np.ndarray[3]) -> None:
        self.att_noise = rot_noise
        self.pos_noise = pos_noise

    def noise_model(self) -> gtsam.noiseModel:
        noise = np.append(self.pos_noise, self.att_noise)
        return gtsam.noiseModel.Diagonal.Sigmas(noise)
    


class CameraNoise:

    def __init__(self, pixel_noise: np.ndarray[2]) -> None:
        self.pixel_noise = pixel_noise
        
    def noise_model(self) -> gtsam.noiseModel:
        noise = self.pixel_noise
        return gtsam.noiseModel.Diagonal.Sigmas(noise)