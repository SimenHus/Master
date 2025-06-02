import pyproj
import numpy as np
from enum import Enum

from src.util import Geometry

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.structs import LLA


class EPSG(Enum):
    GPS = 4326 # WGS84 standard for GPS
    UTM32 = 32632 # EPSG for UTM in trondheim
    LLA = 4979
    ECEF = 4978


# https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates
class Transformation:
    _GPS_to_UTM32 = pyproj.Transformer.from_crs(EPSG.GPS.value, EPSG.UTM32.value, always_xy=True)
    _LLA_to_ECEF = pyproj.Transformer.from_crs(EPSG.LLA.value, EPSG.ECEF.value, always_xy=True)
    @classmethod
    def GPS_to_UTM32(clc, lon, lat) -> 'Geometry.Vector3':
        return np.array(clc._GPS_to_UTM32.transform(lon, lat))

    @classmethod
    def LLA_to_ECEF(clc, lla: 'LLA') -> 'Geometry.Vector3':
        return np.array(clc._LLA_to_ECEF.transform(lla.lon, lla.lat, lla.alt))

    @classmethod
    def ECEF_to_NED(clc, ecef, lla_ref: 'LLA') -> 'Geometry.Vector3':
        ecef_ref = clc.LLA_to_ECEF(lla_ref)
        delta = ecef - ecef_ref
        R = clc.R_NED(lla_ref)
        return R @ delta
    
    @classmethod
    def LLA_to_NED(clc, lla: 'LLA', lla_ref: 'LLA') -> 'Geometry.Vector3':
        ecef = clc.LLA_to_ECEF(lla)
        return clc.ECEF_to_NED(ecef, lla_ref)
    
    @classmethod
    def R_NED(clc, lla: 'LLA') -> 'Geometry.SO3':
        lon, lat = lla.lon * np.pi/180, lla.lat * np.pi/180
        sin = lambda x: np.sin(x)
        cos = lambda x: np.cos(x)
        return np.array([
            [-sin(lat)*cos(lon), -sin(lat)*sin(lon), cos(lat)],
            [-sin(lon), cos(lon), 0],
            [-cos(lat)*cos(lon), -cos(lat)*sin(lon), -sin(lat)]
        ])