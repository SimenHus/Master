
from dataclasses import dataclass
@dataclass
class MapPointObservation:
    timestep: int
    camera_id: int
    keypoint: Keypoint = None

    def __eq__(self, other) -> bool:
        if not isinstance(other, MapPointObservation): return False
        return self.timestep == other.timestep and self.camera_id == other.timestep
    
    def __hash__(self) -> int:
        return hash(tuple(self.timestep, self.camera_id))

@dataclass
class MapPoint:
    id: int
    position: np.ndarray[3]
    descriptor: Descriptor
    observations: set[MapPointObservation] = field(default_factory=set)
    # visible_times = 0
    # matched_times = 0
    # normal_vector = np.zeros(3)
    is_outlier = False

    def add_observation(self, frame: Frame, keypoint: Keypoint):
        self.observations.add(MapPointObservation(frame.timestep, frame.camera_id, keypoint))

    # def compute_descriptor(self, descriptors):
    #     """Average or median of descriptors from all observations"""
    #     self.descriptor = np.mean(descriptors, axis=0)

    # def update_normal(self, frame_pose):
    #     """Update normal vector based on camera viewing direction"""
    #     view_dir = (self.position - frame_pose[:3, 3])
    #     view_dir /= np.linalg.norm(view_dir)
    #     self.normal_vector += view_dir
    #     self.normal_vector /= np.linalg.norm(self.normal_vector)

    def __eq__(self, other) -> bool:
        if not isinstance(other, MapPoint): return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        return self.id
    

class Map:
    pass