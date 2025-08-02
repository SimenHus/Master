

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
    def from_vector(vals: Vector6, radians = False) -> 'SE3':
        if type(vals) == list: vals = np.array(vals)
        if not np.issubdtype(vals.dtype, np.floating):
            vals = vals.astype(float)
        pos = vals[3:]
        rot = SO3.from_vector(vals[:3], radians=radians)
        return SE3(rot, pos)

    @staticmethod
    def as_vector(pose: 'SE3', radians = False) -> Vector6:
        rot = SO3.as_vector(pose.rotation(), radians=radians)
        pos = pose.translation()
        return np.append(rot, pos)
    
    @staticmethod
    def transform_twist(pose: 'SE3', xi: Vector6) -> Vector6:
        return pose.inverse().AdjointMap() @ xi
    
    @staticmethod
    def transform_cov(pose: 'SE3', cov):
        Ad = pose.inverse().AdjointMap()
        return Ad @ cov @ Ad.T
    
    @classmethod
    def NED_to_RDF(clc, pose: 'SE3') -> 'SE3':
        return pose.compose(clc.NED_to_RDF_map())
    
    @classmethod
    def RDF_to_NED(clc, pose: 'SE3') -> 'SE3':
        return pose.compose(clc.RDF_to_NED_map())
    
    @staticmethod
    def NED_to_RDF_map() -> 'SE3':
        rot = SO3.RzRyRx(np.array([90, 0, 90])*np.pi / 180)
        return SE3(rot, [0, 0, 0])
    
    @classmethod
    def RDF_to_NED_map(clc) -> 'SE3':
        return clc.NED_to_RDF_map().inverse()
    
    @classmethod
    def RDF_to_NED_cov(clc, cov):
        return clc.transform_cov(clc.NED_to_RDF_map(), cov)

    @staticmethod
    def align_frames(X: list['SE3'], Y: list['SE3']) -> tuple[float, 'SE3']:
        X = np.array([T.translation() for T in X]).T
        Y = np.array([T.translation() for T in Y]).T

        mu_x = X.mean(axis=1).reshape(-1, 1)
        mu_y = Y.mean(axis=1).reshape(-1, 1)

        var_x = np.square(X - mu_x).sum(axis=0).mean()
        cov_xy = ((Y - mu_y) @ (X - mu_x).T) / X.shape[1]

        U, D, VH = np.linalg.svd(cov_xy)
        S = np.eye(X.shape[0])
        if np.linalg.det(U) * np.linalg.det(VH) < 0:
            S[-1, -1] = -1
        c = np.trace(np.diag(D) @ S) / var_x
        R = U @ S @ VH
        t = mu_y - c * R @ mu_x

        return c, SE3(SO3(R), t.reshape(3,))

class SO3(gtsam.Rot3): # Abstraction of gtsam Rot3 class
    
    @staticmethod
    def from_vector(vals: Vector3, radians = False) -> 'SO3':
        if type(vals) == list: vals = np.array(vals)
        if not np.issubdtype(vals.dtype, np.floating):
            vals = vals.astype(float)
        if not radians: vals *= np.pi/180
        return SO3.RzRyRx(*vals)
    
    @staticmethod
    def as_vector(rot: 'SO3', radians = False) -> Vector3:
        vals = rot.xyz()
        if not radians: vals *= 180 / np.pi
        return vals


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
        return SE3.from_vector(np.append(self.attitude, self.position), radians=True)
    
    @property
    def twist(self) -> Vector6:
        return np.append(self.attrate, self.velocity)
    
    @property
    def sigmas(self) -> Vector6:
        return np.append(self.atterror, self.poserror)