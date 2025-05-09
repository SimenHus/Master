
import cv2
import numpy as np
from src.util import Geometry, DataAssociation

from copy import copy

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.structs import KeyFrame



from dataclasses import dataclass, field
@dataclass
class MapPointObservation:
    timestep: int
    camera_id: int
    keypoint: cv2.KeyPoint = None

    def __eq__(self, other) -> bool:
        if not isinstance(other, MapPointObservation): return False
        return self.timestep == other.timestep and self.camera_id == other.timestep
    
    def __hash__(self) -> int:
        return hash(tuple(self.timestep, self.camera_id))


class MapPoint:
    next_id = 0


    def __init__(self, pos: 'Geometry.Vector3', ref_keyframe: 'KeyFrame', map: 'Map') -> None:
        
        self.observations: dict['KeyFrame': int] = {} # Keyframe ID: observation ID in keyframe
        self.reference_keyframe = ref_keyframe
        self.map = map

        self.set_world_pos(pos)

        self.id = MapPoint.next_id
        MapPoint.next_id += 1


        self.normal_vector = self.get_world_pos() - self.get_reference_keyframe().get_camera_center()
        self.normal_vector = Geometry.Vector3.normalize(self.normal_vector)

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

        dist = Geometry.Vector3.norm(pos - reference_keyframe.get_camera_center())
        level = reference_keyframe.keypoints_und[observations[reference_keyframe]].octave
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


    def as_dict(self) -> dict:
        return {
            'id': self.id,
            'pos': self.get_world_pos().tolist(),
            'observations': {kf.id: index for kf, index in self.get_observations().items()},
            'descriptor': self.descriptor.tolist(),
            'normal_vector': self.get_normal_vector().tolist(),
        }
    

class Map:
    next_id = 0

    def __init__(self, init_keyframe_id: int = 0) -> None:
        self.max_keyframe_id = init_keyframe_id
        self.init_keyframe_id = init_keyframe_id

        self._is_in_use = False

        self.keyframes: set['KeyFrame'] = set()
        self.map_points: set['MapPoint'] = set()
        self.keyframe_origins: set['KeyFrame'] = set()
        self.reference_map_points: set['MapPoint'] = set()

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
    
    def get_all_map_points(self) -> set['MapPoint']: return self.map_points
    def get_all_keyframes(self) -> set['KeyFrame']: return self.keyframes

    def set_reference_map_points(self, map_points: set['MapPoint']) -> None: self.reference_map_points = map_points