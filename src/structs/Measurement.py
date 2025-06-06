
from src.util import Geometry, Time
from src.structs import Frame

from enum import Enum
from dataclasses import dataclass

import numpy as np


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.structs import Frame


@dataclass
class MeasurementBaseClass:
    timestep: int

@dataclass
class VesselMeasurement(MeasurementBaseClass):
    attitude: np.ndarray[3] = np.array([0, 0, 0])
    att_rate: np.ndarray[3] = np.array([0, 0, 0])
    att_error: np.ndarray[3] = np.array([1, 1, 1])*1e6

    position: np.ndarray[3] = np.array([0, 0, 0])
    velocity: np.ndarray[3] = np.array([0, 0, 0])
    acceleration: np.ndarray[3] = np.array([0, 0, 0])
    pos_error: np.ndarray[3] = np.array([1, 1, 1])*1e6

    heave: float = 0

    def as_pose(self) -> Geometry.SE3:
        pos = Geometry.Point3(*self.position)
        rot = Geometry.SO3.Expmap(self.attitude)
        pose = Geometry.SE3(rot, pos)
        return pose
    
    def pose_noise(self) -> Geometry.SE3Noise:
        noise = Geometry.SE3Noise(self.pos_error, self.att_error)
        return noise
    
    @staticmethod
    def from_json(json_dict: dict) -> 'VesselMeasurement':
        vessel_info = json_dict['own_vessel']
        timestep = Time.TimeConversion.UTC_to_POSIX(json_dict['meas_time'])

        return VesselMeasurement(
            attitude = np.array(vessel_info['attitude']) * np.pi/180,
            att_rate = np.array(vessel_info['attrate']) * np.pi/180,
            att_error = np.array(vessel_info['atterror']) * np.pi/180,
            position = np.array(vessel_info['position']),
            velocity = np.array(vessel_info['velocity']),
            acceleration = np.array(vessel_info['acceleration']),
            pos_error = np.array(vessel_info['poserror']),
            heave = vessel_info['heave'],
            timestep = timestep
        )

@dataclass
class CameraMeasurement(MeasurementBaseClass):
    camera_id: int
    frame: Frame
    latest_vessel_measurement: VesselMeasurement

@dataclass
class OdometryMeasurement(VesselMeasurement):
    pass


class MeasurementType(Enum):
    UNKNOWN = 0
    VESSEL = 1
    CAMERA = 2
    ODOMETRY = 3

class MeasurementIdentifier:
    @staticmethod
    def identify(measurement: MeasurementBaseClass) -> MeasurementType:
        
        if type(measurement) == CameraMeasurement: return MeasurementType.CAMERA
        if type(measurement) == VesselMeasurement: return MeasurementType.VESSEL
        if type(measurement) == OdometryMeasurement: return MeasurementType.ODOMETRY
        return MeasurementType.UNKNOWN
