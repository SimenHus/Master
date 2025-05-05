
from src.structs import Frame, KeyFrame, Camera
# from src.backend import LoopClosing
# from src.mapping import LocalMapping
from src.atlas import Atlas
from src.util import Geometry, DataAssociation

from enum import Enum
import cv2


class TrackerState(Enum):
    SYSTEM_NOT_READY = -1
    NO_IMAGES_YET = 0
    NOT_INITIALIZED = 1
    OK = 2

# https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/src/Tracking.cc#L1794
# and https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/include/Tracking.h#L138
class Tracker:
    current_frame: Frame
    last_frame: Frame

    first_frame_id: int
    initial_frame_id: int
    last_init_frame_id: int
    
    t0: int

    # Pointers to other threads
    # loop_closing: LoopClosing
    # local_mapper: LocalMapping

    atlas: Atlas
    created_map: bool
    map_updated: bool

    state: int
    last_processed_state: int

    last_id: int
    init_id: int

    reference_KF: KeyFrame


    camera = Camera() # More logic around camera should be added

    # Monocular initialization variables
    init_last_matches: list[int]
    init_matches: list[cv2.DMatch]
    prev_matched: list[Geometry.Point2]
    init_P3D: list[Geometry.Point3]
    initial_frame: Frame
    ready_to_initialize: bool = False
    set_init: bool
    
    def track(self) -> None:
        current_map = self.atlas.get_current_map()
        if not current_map: print('ERROR: No active map in atlas')

        if self.state != TrackerState.NO_IMAGES_YET:
            if self.last_frame.timestep > self.current_frame.timestep: # Sanity check for timesteps
                print('Error: Frame with timestep older than previous frame detected')
                return

        if self.state == TrackerState.NO_IMAGES_YET: self.state = TrackerState.NOT_INITIALIZED

        self.last_processed_state = self.state # Update previous tracker state


        if self.state == TrackerState.NOT_INITIALIZED:
            self.monocular_initialization()
            if self.state != TrackerState.OK: # Not properly initialized
                self.last_frame = Frame.copy(self.current_frame)
                return
            
            self.first_frame_id = self.current_frame.id
            return

        # If system is not initialized, it will return before this statement
        ok: bool = None # System is initialized

        # Handle tracking / local mapping mode. For now, only tracking is available
        if self.state == TrackerState.LOST:
            ok = self.relocalization() # Perform some relocalization procedure


        if not self.current_frame.reference_KF: # Check if current frame is keyframe
            self.current_frame.reference_KF = self.reference_KF # If current frame not keyframe, update with tracker keyframe

        ok = self.track_local_map()

        if ok: self.state = TrackerState.OK

        need_KF = self.need_new_keyframe()

        if need_KF and ok:
            self.create_new_keyframe()

        if not self.current_frame.reference_KF: # Check if current frame is keyframe
            self.current_frame.reference_KF = self.reference_KF # If current frame not keyframe, update with tracker keyframe

        self.last_frame = Frame.copy(self.current_frame)



    def monocular_initialization(self) -> None:
        
        n_features_init_threshold = 100
        # Check if we have enough keypoints to initialize
        if len(self.current_frame.keypoints) < n_features_init_threshold: return


        if not self.ready_to_initialize:
            self.initial_frame = Frame.copy(self.current_frame)
            self.last_frame = Frame.copy(self.current_frame)
            self.prev_matched = [key_und.pt for key_und in self.current_frame.keypoints_und]

            # fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            self.ready_to_initialize = True
            return
        
        # We are ready to initialize if we reach this point
        matcher = DataAssociation.Matcher(0.9, True)
        self.init_matches = matcher.search_for_initialization(self.initial_frame, self.current_frame, self.prev_matched)

        if self.init_matches < 100: # Not enough matches
            self.ready_to_initialize = False
            return
        
        # Attempt to reconstruct a scenario using two images. This will be the initial map
        success, Tcw, self.init_P3D = self.camera.reconstruct_with_two_views(self.initial_frame.keypoints_und, self.current_frame.keypoints_und, self.init_matches)
        if not success: return
            
        self.initial_frame.set_pose(Geometry.SE3()) # Initialize the first point at the origin
        self.current_frame.set_pose(Tcw) # Initialize first movement to be the second frame and the estimated movement
        self.create_initial_map_monocular()


    def create_initial_map_monocular(self) -> None:
        pass

    def create_map_in_atlas(self) -> None:
        pass # Logic to create a new map in atlas

    def create_new_keyframe(self) -> None:
        pass # New keyframe logic

    def need_new_keyframe(self) -> bool:
        return False

    def track_local_map(self) -> bool:
        return False

    def track_reference_keyframe(self) -> bool:
        return False

    def relocalization(self) -> bool:
        return False # Perform relocalization

    def grab_image_monocular(self, image: cv2.Mat, timestep: int) -> Geometry.SE3:

        # Process image in necessary ways (grayscale etc)
        frame = image

        # Maybe perform some checks if system is initialized?
        # if self.state == TrackerState.NOT_INITIALIZED or self.state == TrackerState.NO_IMAGES_YET:
        self.current_frame = Frame(frame, timestep)

        if self.state == TrackerState.NO_IMAGES_YET:
            self.t0 = timestep

        self.last_id = self.current_frame.id
        self.track()

        return self.current_frame.get_pose()
