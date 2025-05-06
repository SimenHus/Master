
from src.structs import Map
from src.util import Logging


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.structs import KeyFrame

class Atlas:
    logger = Logging.get_logger('Atlas')

    def __init__(self, init_keyframe_id: int = None) -> None:
        self.last_init_keyframe_id_map = init_keyframe_id

        self.current_map: Map = None
        self.maps: set[Map] = set()

    def get_current_map(self) -> Map | None:
        if not self.current_map: self.create_new_map()
        return self.current_map


    def create_new_map(self) -> None:
        
        self.logger.info(f'Creating new map with ID: {Map.next_id}')

        if self.current_map:
            if len(self.maps) > 0 and self.last_init_keyframe_id_map < self.current_map.get_max_keyframe_id():
                self.last_init_keyframe_id_map = self.current_map.get_max_keyframe_id() + 1 # Initial keyframe of new map is one higher than maximum keyframe index of previous map
            self.current_map.set_stored_map() # Flag current map as deactive
            self.logger.info(f'Stored map with ID: {self.current_map.get_id()}')

        self.logger.info(f'Created new map with last keyframe ID: {self.last_init_keyframe_id_map}')
        self.current_map = Map(self.last_init_keyframe_id_map)
        self.current_map.set_current_map() # Flag new map as active
        self.maps.add(self.current_map)

    def add_keyframe(self, keyframe: 'KeyFrame') -> None:
        map = keyframe.get_map()