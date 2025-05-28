

import gtsam
import numpy as np
from dataclasses import dataclass

class Point3:
    pass

class Point2:
    pass

class Vector:
    @staticmethod
    def normalize(vector: 'Vector') -> 'Vector':
        """Returns normalized vector"""
        return vector / np.linalg.norm(vector)

    @staticmethod
    def norm(vector: 'Vector') -> float:
        """Returns norm of vector"""
        return np.linalg.norm(vector)


class Vector3(Vector):
    pass

class Vector6(Vector):
    pass


class SE3(gtsam.Pose3): # Abstraction of the gtsam Pose3 class
    
    @staticmethod
    def from_vector(vals: Vector6, radians = True) -> 'SE3':
        pos = vals[3:]
        if not radians: vals[:3] *= np.pi/180
        rot = SO3.RzRyRx(*vals[:3])

        return SE3(rot, pos)

    @staticmethod
    def as_vector(pose: 'SE3', show_degrees = True) -> Vector6:
        rot = pose.rotation().rpy()
        if show_degrees: rot *= 180 / np.pi
        pos = pose.translation()
        return np.append(rot, pos)
    
    @staticmethod
    def transform_twist(pose: 'SE3', xi: Vector6) -> Vector6:
        return pose.inverse().AdjointMap() @ xi

class SO3(gtsam.Rot3): # Abstraction of gtsam Rot3 class
    pass


@dataclass
class State:
    position: Vector3 = np.array([0, 0, 0])
    velocity: Vector3 = np.array([0, 0, 0])
    acceleration: Vector3 = np.array([0, 0, 0])
    poserror: Vector3 = np.array([0, 0, 0])

    attitude: Vector3 = np.array([0, 0, 0])
    attrate: Vector3 = np.array([0, 0, 0])
    atterror: Vector3 = np.array([0, 0, 0])

    @property
    def pose(self) -> SE3:
        return SE3.from_vector(np.append(self.attitude, self.position))
    
    @property
    def twist(self) -> Vector6:
        return np.append(self.attrate, self.velocity)
    
    @property
    def sigmas(self) -> Vector6:
        return np.append(self.atterror, self.poserror)