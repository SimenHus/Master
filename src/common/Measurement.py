
from src.motion import Pose3, Rot3, Point3, Pose3Noise
from dataclasses import dataclass

import numpy as np

@dataclass
class VesselMeasurement:
    attitude: np.ndarray[3]
    att_rate: np.ndarray[3]
    att_error: np.ndarray[3]

    position: np.ndarray[3]
    velocity: np.ndarray[3]
    acceleration: np.ndarray[3]
    pos_error: np.ndarray[3]

    heave: float
    timestamp: str

    def as_pose(self) -> Pose3:
        pos = Point3(*self.position)
        rot = Rot3.RzRyRx(*self.attitude)
        pose = Pose3(rot, pos)
        return pose
    
    def pose_noise(self) -> Pose3Noise:
        noise = Pose3Noise(self.pos_error, self.att_error)
        return noise
    
    @staticmethod
    def from_json(json_dict: dict) -> 'VesselMeasurement':
        vessel_info = json_dict['own_vessel']

        return VesselMeasurement(
            attitude = vessel_info['attitude'],
            att_rate = vessel_info['attrate'],
            att_error = vessel_info['atterror'],
            position = vessel_info['position'],
            velocity = vessel_info['velocity'],
            acceleration = vessel_info['acceleration'],
            pos_error = vessel_info['poserror'],
            heave = vessel_info['heave'],
            timestamp = json_dict['meas_time']
        )

    
@dataclass
class CameraMeasurement:
    image: np.ndarray
    latest_vessel_measurement: VesselMeasurement
    timestamp: str