
from src.structs import Map
from src.util import Logging


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.structs import KeyFrame, MapPoint, MapPointDB

class Atlas:
    logger = Logging.get_logger('Atlas')

    def __init__(self, init_keyframe_id: int = 0) -> None:
        self.last_init_keyframe_id_map = init_keyframe_id

        self.__current_map: Map = None
        self.maps: set[Map] = set()

    def get_current_map(self) -> Map | None:
        if not self.__current_map: self.create_new_map()
        return self.__current_map

    def add_map_point(self, map_point: 'MapPoint') -> None:
        map_point.get_map().add_map_point(map_point)

    def create_new_map(self) -> None:
        
        self.logger.info(f'Creating new map with ID: {Map.next_id}')


        if self.__current_map:
            if len(self.maps) > 0 and self.last_init_keyframe_id_map < self.__current_map.get_max_keyframe_id():
                self.last_init_keyframe_id_map = self.__current_map.get_max_keyframe_id() + 1 # Initial keyframe of new map is one higher than maximum keyframe index of previous map
            self.__current_map.set_stored_map() # Flag current map as deactive
            self.logger.info(f'Stored map with ID: {self.__current_map.get_id()}')

        self.logger.info(f'Created new map with last keyframe ID: {self.last_init_keyframe_id_map}')
        self.__current_map = Map(self.last_init_keyframe_id_map)
        self.__current_map.set_current_map() # Flag new map as active
        self.maps.add(self.__current_map)

    def add_keyframe(self, keyframe: 'KeyFrame') -> None:
        keyframe.get_map().add_keyframe(keyframe)

    def map_points_in_map(self) -> int:
        return self.get_current_map().map_points_in_map()
    
    def get_all_map_points(self) -> 'MapPointDB': return self.get_current_map().get_all_map_points()

    def set_reference_map_points(self, map_points: 'MapPointDB') -> None: self.get_current_map().set_reference_map_points(map_points)

    def get_all_keyframes(self) -> set['KeyFrame']: return self.get_current_map().get_all_keyframes()