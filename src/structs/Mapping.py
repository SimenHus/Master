
import cv2
import numpy as np
from src.util import Geometry

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
        
        self.observations: dict[int: int] = {} # Keyframe ID: observation ID in keyframe
        self.reference_keyframe = ref_keyframe
        self.map = map

        self.set_world_pos(pos)

        self.id = MapPoint.next_id
        MapPoint.next_id += 1


        self.normal_vector = self.get_world_pos() - self.get_reference_keyframe().get_camera_center()
        self.normal_vector = Geometry.Vector3.normalize(self.normal_vector)


    def set_world_pos(self, pos: 'Geometry.Point3') -> None: self.world_pos = pos

    def get_world_pos(self) -> 'Geometry.Point3': return self.world_pos

    def get_normal_vector(self) -> 'Geometry.Vector3': return self.normal_vector

    def set_normal_vector(self, vec: 'Geometry.Vector3'): self.normal_vector = vec

    def get_reference_keyframe(self) -> 'KeyFrame': return self.reference_keyframe

    def get_observations(self) -> dict: return self.observations

    def add_observation(self, keyframe: 'KeyFrame', id: int) -> None: self.observations[keyframe.id] = id

    def update_normal_and_depth(self) -> None: pass
#         {
#     map<KeyFrame*,tuple<int,int>> observations;
#     KeyFrame* pRefKF;
#     Eigen::Vector3f Pos;
#     {
#         unique_lock<mutex> lock1(mMutexFeatures);
#         unique_lock<mutex> lock2(mMutexPos);
#         if(mbBad)
#             return;
#         observations = mObservations;
#         pRefKF = mpRefKF;
#         Pos = mWorldPos;
#     }

#     if(observations.empty())
#         return;

#     Eigen::Vector3f normal;
#     normal.setZero();
#     int n=0;
#     for(map<KeyFrame*,tuple<int,int>>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
#     {
#         KeyFrame* pKF = mit->first;

#         tuple<int,int> indexes = mit -> second;
#         int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);

#         if(leftIndex != -1){
#             Eigen::Vector3f Owi = pKF->GetCameraCenter();
#             Eigen::Vector3f normali = Pos - Owi;
#             normal = normal + normali / normali.norm();
#             n++;
#         }
#         if(rightIndex != -1){
#             Eigen::Vector3f Owi = pKF->GetRightCameraCenter();
#             Eigen::Vector3f normali = Pos - Owi;
#             normal = normal + normali / normali.norm();
#             n++;
#         }
#     }

#     Eigen::Vector3f PC = Pos - pRefKF->GetCameraCenter();
#     const float dist = PC.norm();

#     tuple<int ,int> indexes = observations[pRefKF];
#     int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
#     int level;
#     if(pRefKF -> NLeft == -1){
#         level = pRefKF->mvKeysUn[leftIndex].octave;
#     }
#     else if(leftIndex != -1){
#         level = pRefKF -> mvKeys[leftIndex].octave;
#     }
#     else{
#         level = pRefKF -> mvKeysRight[rightIndex - pRefKF -> NLeft].octave;
#     }

#     //const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;
#     const float levelScaleFactor =  pRefKF->mvScaleFactors[level];
#     const int nLevels = pRefKF->mnScaleLevels;

#     {
#         unique_lock<mutex> lock3(mMutexPos);
#         mfMaxDistance = dist*levelScaleFactor;
#         mfMinDistance = mfMaxDistance/pRefKF->mvScaleFactors[nLevels-1];
#         mNormalVector = normal/n;
#     }
# }
    # def compute_descriptor(self, descriptors):
    #     """Average or median of descriptors from all observations"""
    #     self.descriptor = np.mean(descriptors, axis=0)


    # def __eq__(self, other) -> bool:
    #     if not isinstance(other, MapPoint): return False
    #     return self.id == other.id
    
    # def __hash__(self) -> int:
    #     return self.id
    

class Map:
    next_id = 0

    def __init__(self, init_keyframe_id: int = 0) -> None:
        self.init_keyframe_id = init_keyframe_id
        self.max_keyframe_id = init_keyframe_id

        self._is_in_use = False

        self.map_points: set['MapPoint'] = set()
        self.keyframes: set['KeyFrame'] = set()

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