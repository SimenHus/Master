
import cv2
import numpy as np
from src.util import Geometry, DataAssociation

from copy import copy

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.structs import KeyFrame


class MapPointDB:

    def __init__(self) -> None:
        self._map_points: dict[int, 'MapPoint'] = {} # MapPointID: MapPoint
        self._outlier_ids: set[int] = set()

    def add(self, map_point: 'MapPoint') -> None:
        if map_point.id not in self._map_points.keys():
            self[map_point.id] = map_point
            if map_point.is_outlier(): self.set_outlier(map_point.id)

    def update_map_point(self, map_point: 'MapPoint') -> None:
        self[map_point] = map_point

    def size(self) -> int:
        return self.__len__()
    
    def clear(self) -> None:
        self._map_points = {}
        self._outlier_ids = set()
    
    def set_outlier(self, id: 'int | MapPoint') -> None:
        """Set supplied ID/MapPoint as outlier"""
        if isinstance(id, int): self._outlier_ids.add(id)
        if isinstance(id, MapPoint): self._outlier_ids.add(id.id)
        
        self[id].set_outlier()
    
    def get_outliers(self) -> 'MapPointDB':
        sub_database = MapPointDB()
        for outlier_id in self._outlier_ids:
            sub_database.add(self[outlier_id])
            sub_database.set_outlier(outlier_id)
        return sub_database
    
    def get_outlier_ids(self) -> set[int]:
        return self._outlier_ids
    
    def __len__(self) -> int:
        return len(self._map_points)
    
    def __setitem__(self, key: 'int | MapPoint', value: 'MapPoint') -> None:
        if isinstance(key, int):
            self._map_points[key] = value
            return
        if isinstance(key, MapPoint):
            for id, _map_point in self._map_points.items():
                if _map_point == value:
                    self._map_points[id] = value
                    return
        raise KeyError(f'MapPoint with not found in database')


    def __getitem__(self, key: 'int | MapPoint') -> 'MapPoint':
        if isinstance(key, int): return self._map_points[key]
        if isinstance(key, MapPoint):
            for _map_point in self._map_points.values():
                if _map_point == key: return _map_point
        raise KeyError
    
    def __iter__(self):
        return iter(self._map_points.values())

    def __contains__(self, identifier: 'MapPoint | int') -> bool:
        """Check if supplied map point / id exists in database"""

        
        if isinstance(identifier, int): return identifier in self._map_points.keys()
        if isinstance(identifier, MapPoint):
            for _map_point in self._map_points.values(): # Loop through existing map points
                if _map_point == identifier: return True # Check if MapPoint is similar to existing mappoints
        return False


class MapPoint:
    next_id = 0


    def __init__(self, pos: 'Geometry.Vector3', ref_keyframe: 'KeyFrame', map: 'Map') -> None:
        
        self.observations: dict['KeyFrame': int] = {} # Keyframe ID: observation ID in keyframe
        self.reference_keyframe = ref_keyframe
        self.map = map

        self.set_world_pos(pos)

        self.id = MapPoint.next_id
        MapPoint.next_id += 1

        self._outlier = False
        self.normal_vector = self.get_world_pos() - self.get_reference_keyframe().get_camera_center()
        self.normal_vector = Geometry.Vector3.normalize(self.normal_vector)

    def set_outlier(self) -> None: self._outlier = True
    def set_inlier(self) -> None: self._outlier = False
    def is_outlier(self) -> bool: return self._outlier

    def get_map(self) -> 'Map': return self.map

    def set_world_pos(self, pos: 'Geometry.Point3') -> None: self.world_pos = pos

    def get_world_pos(self) -> 'Geometry.Point3': return self.world_pos

    def get_normal_vector(self) -> 'Geometry.Vector3': return self.normal_vector

    def set_normal_vector(self, vec: 'Geometry.Vector3'): self.normal_vector = vec

    def get_reference_keyframe(self) -> 'KeyFrame': return self.reference_keyframe

    def get_observations(self) -> dict: return self.observations

    def add_observation(self, keyframe: 'KeyFrame', id: int) -> None: self.observations[keyframe] = id

    def update_normal_and_depth(self) -> None:
        observations = self.observations
        pos = self.world_pos
        reference_keyframe = self.reference_keyframe

        if len(observations) == 0: return

        normal = np.zeros((3,))

        # Loop through all observations of MapPoint and create an averaged normal
        for keyframe in observations.keys():
            Ow_i = keyframe.get_camera_center()
            normal_i = pos - Ow_i
            normal = normal + Geometry.Vector3.normalize(normal_i)

        normal = normal / len(observations)

        # dist = Geometry.Vector3.norm(pos - reference_keyframe.get_camera_center())
        # level = reference_keyframe.keypoints_und[observations[reference_keyframe]].octave
        # level_scale_factor = reference_keyframe.scale_factors[level]

        # self.max_distance = dist * level_scale_factor
        # self.min_distance = self.max_distance / reference_keyframe.scale_factors[-1]
        self.normal_vector = normal

    def compute_distinct_descriptor(self) -> None:
        """Loop through descriptors of all observations of this map point.
        The descriptor with the lowest median to other points is chosen as descriptor"""
        observations = self.observations
        N = len(observations)
        if N == 0: return

        descriptors = [kf.descriptors[index] for kf, index in observations.items()]

        distances = np.zeros((N-1, N-1))
        for i, desc_i in enumerate(descriptors[:-1]):
            for j, desc_j in enumerate(descriptors[i+1:]):
                dist_ij = DataAssociation.Matcher.descriptor_distance(desc_i, desc_j)
                distances[i, j] = dist_ij
                distances[j, i] = dist_ij

        best_median = -1
        best_id = -1

        for i in range(N-1):
            dists = distances[i, :]
            dists = np.sort(dists)
            median = np.median(dists)
            if median < best_median or best_median == -1:
                best_median = median
                best_id = i
        self.descriptor = copy(descriptors[best_id])

    def __eq__(self, other) -> bool:
        if not isinstance(other, MapPoint): return False
        return self.id == other.id or DataAssociation.Matcher.is_match(self.descriptor, other.descriptor)

    def as_dict(self) -> dict:
        return {
            'id': self.id,
            'pos': self.get_world_pos().tolist(),
            'observations': {kf.id: index for kf, index in self.get_observations().items()},
            'descriptor': self.descriptor.tolist(),
            'normal_vector': self.get_normal_vector().tolist(),
            'outlier': self.is_outlier()
        }
    

class Map:
    next_id = 0

    def __init__(self, init_keyframe_id: int = 0) -> None:
        self.max_keyframe_id = init_keyframe_id
        self.init_keyframe_id = init_keyframe_id

        self._is_in_use = False

        self.keyframes: set['KeyFrame'] = set()
        self.map_points: MapPointDB = MapPointDB()
        self.keyframe_origins: set['KeyFrame'] = set()
        self.reference_map_points: MapPointDB = MapPointDB()

        self.id = Map.next_id
        Map.next_id += 1


    def add_keyframe(self, keyframe: 'KeyFrame') -> None:
        if len(self.keyframes) == 0:
            self.init_keyframe_id = keyframe.id
            self.init_keyframe = keyframe
            self.keyframe_lower_id = keyframe
        self.keyframes.add(keyframe)
        
        if keyframe.id > self.max_keyframe_id: self.max_keyframe_id = keyframe.id
        if keyframe.id < self.keyframe_lower_id.id: self.keyframe_lower_id = keyframe

    def add_map_point(self, map_point: 'MapPoint') -> None: self.map_points.add(map_point)

    def get_original_keyframe(self) -> 'KeyFrame':
        return self.init_keyframe

    def get_max_keyframe_id(self) -> int:
        return self.max_keyframe_id

    def is_in_use(self) -> bool:
        return self._is_in_use
    
    def clear(self) -> None:
        self.max_keyframe_id = self.init_keyframe_id

    def set_current_map(self) -> None:
        self._is_in_use = True

    def set_stored_map(self) -> None:
        self._is_in_use = False

    def get_id(self) -> int:
        return self.id
    
    def map_points_in_map(self) -> int:
        return len(self.map_points)
    
    def get_all_map_points(self) -> MapPointDB: return self.map_points
    def get_all_keyframes(self) -> set['KeyFrame']: return self.keyframes

    def set_reference_map_points(self, map_points: MapPointDB) -> None: self.reference_map_points = map_points