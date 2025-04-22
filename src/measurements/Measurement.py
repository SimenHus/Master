
from src.motion import Pose3, Rot3, Point3, Pose3Noise
from src.common import Frame

from enum import Enum
from dataclasses import dataclass

import numpy as np

@dataclass
class MeasurementBaseClass:
    timestamp: str

@dataclass
class VesselMeasurement(MeasurementBaseClass):
    attitude: np.ndarray[3]
    att_rate: np.ndarray[3]
    att_error: np.ndarray[3]

    position: np.ndarray[3]
    velocity: np.ndarray[3]
    acceleration: np.ndarray[3]
    pos_error: np.ndarray[3]

    heave: float

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
class CameraMeasurement(MeasurementBaseClass):
    camera_id: int
    frame: Frame
    latest_vessel_measurement: VesselMeasurement


class MeasurementType(Enum):
    UNKNOWN = 0
    VESSEL = 1
    CAMERA = 2

class MeasurementIdentifier:
    @staticmethod
    def identify(measurement: MeasurementBaseClass) -> MeasurementType:
        
        if type(measurement) == CameraMeasurement: return MeasurementType.CAMERA
        if type(measurement) == VesselMeasurement: return MeasurementType.VESSEL
        return MeasurementType.UNKNOWN
