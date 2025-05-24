
from src.structs import Frame, KeyFrame, Camera, MapPoint, MapPointDB
# from src.backend import LoopClosing
# from src.mapping import LocalMapping
from src.atlas import Atlas
from src.util import Geometry, DataAssociation, Logging

from copy import copy

from enum import Enum
import cv2


class TrackerState(Enum):
    SYSTEM_NOT_READY = -1
    NO_IMAGES_YET = 0
    NOT_INITIALIZED = 1
    OK = 2


class Tracker:
    logger = Logging.get_logger('Tracker')

    # Pointers to other threads
    # loop_closing: LoopClosing
    # local_mapper: LocalMapping

    first_frame_id = 0

    # Monocular initialization variables
    required_features_for_init = 100
    required_matches_for_init = 50
    init_matches: list[cv2.DMatch]
    init_P3D: list[Geometry.Point3]
    init_mask: list[bool]
    initial_frame: Frame
    ready_to_init: bool = False

    def __init__(self, atlas: Atlas) -> None:
        self.init_id, self.last_id = 0, 0
        self.state = TrackerState.NO_IMAGES_YET
        self.atlas = atlas

        self.init_extractor = DataAssociation.Extractor()
        self.mono_extractor = DataAssociation.Extractor()

        self.local_keyframes: set[KeyFrame] = set()
        self.local_map_points = MapPointDB()

    def set_camera(self, camera: Camera) -> None:
        self.camera = camera
    
    def track(self) -> None:
        current_map = self.atlas.get_current_map() # Check if atlas contains a map, if not it should be created
        if not current_map:
            self.logger.error('No active map found in atlas')
            return

        # Sanity check if tracking has begun
        if self.state != TrackerState.NO_IMAGES_YET:
            if self.last_frame.timestep > self.current_frame.timestep: # Sanity check for timesteps
                self.logger.error('Frame with timestep older than previous frame detected')
                return

        # Update state if tracker has received first image
        if self.state == TrackerState.NO_IMAGES_YET: self.state = TrackerState.NOT_INITIALIZED

        # Initial state check and update complete, proceed with tracking
        self.last_processed_state = self.state # Store current state before updated by tracking


        if self.state == TrackerState.NOT_INITIALIZED:
            self.monocular_initialization()
            if self.state != TrackerState.OK: # Not properly initialized
                self.last_frame = Frame.clone(self.current_frame)
                return
            
            self.first_frame_id = self.current_frame.id
            return
        
        # --------------------------------------------------------------------------
        # If system is not initialized, it will return before this statement
        ok: bool = False # Variable to keep track of status of steps in current tracking cycle

        if not self.current_frame.reference_keyframe: # Check if reference keyframe exists for current frame
            self.current_frame.reference_keyframe = self.reference_keyframe # If no reference keyframe found, set tracker keyframe as reference

        # Track and update current frame based on reference keyframe
        self.logger.info(f'Tracking frame with ID/Timestep: {self.current_frame.id} / {self.current_frame.timestep}')
        ok = self.track_reference_keyframe()

        # Frame tracking is ok, track map
        if ok: ok = self.track_local_map()

        if ok:
            self.logger.info(f'Local tracking for frame ID {self.current_frame.id} OK')
            self.state = TrackerState.OK
        else:
            self.logger.info(f'Local tracking for frame ID {self.current_frame.id} NOT OK')

        need_KF = self.need_new_keyframe()

        if need_KF and ok:
            self.logger.info(f'New keyframe required, creating keyframe ID {KeyFrame.next_id} from frame ID {self.current_frame.id}')
            self.create_new_keyframe()

        self.last_frame = Frame.clone(self.current_frame)


    def track_reference_keyframe(self) -> bool:
        """Function to track current frame using reference keyframe"""
        # self.current_frame.compute_BoW()
        reference_keyframe = self.reference_keyframe
        current_frame = self.current_frame

        matcher = DataAssociation.Matcher()
        matches = matcher.search_for_initialization(reference_keyframe, current_frame)
        success, Tcr, P3Ds, mask = self.camera.reconstruct_with_two_views(reference_keyframe.keypoints_normed, current_frame.keypoints_normed, matches)
        if not success: return # Unsuccessfull reconstruction of new keyframe

        matches = [match for i, match in enumerate(matches) if mask[i]] # Get inliers
        P3Ds = [P3D for i, P3D in enumerate(P3Ds) if mask[i]] # Get inliers
        
        current_frame.set_pose(reference_keyframe.get_pose().compose(Tcr))
        all_map_points = self.atlas.get_all_map_points()

        # Update map points in current frame
        # As it stands, this logic should be placed elsewhere
        for i, (match, P3D) in enumerate(zip(matches, P3Ds)):
            map_point = MapPoint(P3D, reference_keyframe, self.atlas.get_current_map())
            map_point.set_init_descriptor(reference_keyframe.descriptors[match.queryIdx])

            # If map point already exists, overwrite map point with global reference
            # If it does not exist, we add it to the atlas active map
            if map_point in all_map_points: map_point = all_map_points[map_point] # Get reference
            elif map_point in reference_keyframe.get_map_points(): map_point = reference_keyframe.get_map_points() # Get reference
            else: self.atlas.add_map_point(map_point) # Does not already exist, update atlas

            if map_point not in reference_keyframe.get_map_points():
                reference_keyframe.add_map_point(map_point, match.queryIdx)
                map_point.add_observation(reference_keyframe, match.queryIdx)

            # Add map point to current frame
            current_frame.add_map_point(map_point, match.trainIdx)
            map_point.add_temp_observation(current_frame, match.trainIdx)

            map_point.compute_distinct_descriptor() # Create/Update distinct descriptor
            map_point.update_normal_and_depth()


        # Perform pose optimization to get better pose estimate???

        match_threshold = 1
        return len(matches) >= match_threshold

    def monocular_initialization(self) -> None:
        
        self.logger.info('Attempting to initialize monocular tracking')
        # Check if we have enough keypoints to initialize
        if len(self.current_frame.keypoints) < self.required_features_for_init:
            self.logger.error(f'Not enough keypoints to initialize: {len(self.current_frame.keypoints)}/{self.required_features_for_init}')
            return

        if not self.ready_to_init:
            self.initial_frame = Frame.clone(self.current_frame)
            self.last_frame = Frame.clone(self.current_frame)
            self.prev_matched = [key_norm.pt for key_norm in self.current_frame.keypoints_normed]

            self.ready_to_init = True
            self.logger.info('Monocular initialization ready')
            return
        
        # We are ready to initialize if we reach this point
        matcher = DataAssociation.Matcher()
        self.init_matches = matcher.search_for_initialization(self.initial_frame, self.current_frame)
        
        if len(self.init_matches) < self.required_matches_for_init: # Not enough matches
            self.ready_to_init = False
            self.logger.error(f'Monocular initialization failed, not enough matches: {len(self.init_matches)}/{self.required_matches_for_init}')
            return
        
        # Attempt to reconstruct a scenario using two images. This will be the initial map
        success, Tcw, self.init_P3D, self.init_mask = self.camera.reconstruct_with_two_views(self.initial_frame.keypoints_normed, self.current_frame.keypoints_normed, self.init_matches)
        if not success: return
        
        self.logger.info('Monocular initialization successfull')
        self.initial_frame.set_pose(Geometry.SE3()) # Initialize the first point at the origin
        self.current_frame.set_pose(Tcw) # Initialize first movement to be the second frame and the estimated movement
        self.create_initial_map_monocular()


    def create_initial_map_monocular(self) -> None:
        self.logger.info('Creating initial map for monocular tracking')
        initial_keyframe = KeyFrame(self.initial_frame, self.atlas.get_current_map())
        current_keyframe = KeyFrame(self.current_frame, self.atlas.get_current_map())


        self.atlas.add_keyframe(initial_keyframe)
        self.atlas.add_keyframe(current_keyframe)
        for i, (match, P3D) in enumerate(zip(self.init_matches, self.init_P3D)):
            # Create MapPoint
            map_point = MapPoint(P3D, current_keyframe, self.atlas.get_current_map())
            
            initial_keyframe.add_map_point(map_point, match.queryIdx)
            current_keyframe.add_map_point(map_point, match.trainIdx)

            map_point.add_observation(initial_keyframe, match.queryIdx)
            map_point.add_observation(current_keyframe, match.trainIdx)

            map_point.compute_distinct_descriptor()
            map_point.update_normal_and_depth()

            self.current_frame.add_map_point(map_point, match.trainIdx)

            self.atlas.add_map_point(map_point)

        self.logger.info(f'New map created with {self.atlas.map_points_in_map()} points')
        

        self.current_frame.set_pose(current_keyframe.get_pose())
        self.last_keyframe_id = self.current_frame.id
        self.last_keyframe = current_keyframe
        
        self.local_keyframes.add(initial_keyframe)
        self.local_keyframes.add(current_keyframe)
        self.local_map_points = copy(self.atlas.get_all_map_points())
        self.reference_keyframe = current_keyframe
        self.current_frame.reference_keyframe = current_keyframe
        
        self.last_frame = Frame.clone(self.current_frame)

        self.atlas.set_reference_map_points(self.local_map_points)
        self.atlas.get_current_map().keyframe_origins.add(initial_keyframe)
        self.state = TrackerState.OK
        self.init_id = current_keyframe.id


    def create_map_in_atlas(self) -> None:
        self.logger.info('Creating a new map in the atlas')
        self.atlas.create_new_map()

        self.last_init_frame_id = self.current_frame.id
        self.init_frame_id = self.current_frame.id + 1
        self.set_init = False
        self.state = TrackerState.NO_IMAGES_YET
        self.ready_to_init = False

        self.last_keyframe = None
        self.reference_keyframe = None

        self.last_frame = None
        self.current_frame = None
        self.init_matches = []

        self.created_map = True


    def create_new_keyframe(self) -> None:
        keyframe = KeyFrame(self.current_frame, self.atlas.get_current_map())

        self.reference_keyframe = keyframe
        self.current_frame.reference_keyframe = keyframe
        if self.last_keyframe:
            keyframe.previous_keyframe = self.last_keyframe
            self.last_keyframe.next_keyframe = keyframe
        else:
            self.logger.error('No previous keyframe existing when attempting to create new')

        self.last_keyframe_id = self.current_frame.id
        self.last_keyframe = keyframe

        # Verify keyframe observation / map points match
        for map_point in keyframe.get_map_points():
            if keyframe not in map_point.get_observations():
                map_point.convert_observation(self.current_frame, keyframe)
                map_point.compute_distinct_descriptor()

        self.atlas.add_keyframe(keyframe) # May need to be moved / removed


    def need_new_keyframe(self) -> bool:
        if Frame.next_id - self.last_keyframe_id > 3: return True
        return False


    def update_local_keyframes(self) -> None:
        
        keyframe_counter: dict['KeyFrame': int] = {} # Keyframe: number of common map points

        for map_point in self.last_frame.get_map_points(): # Loop through map points in previous frame
            observations = map_point.get_observations() # Get map point observations (dict, keyframe: map point id in keyframe)
            for keyframe in observations.keys(): # Loop through keyframes where the map point is observed
                if not keyframe in keyframe_counter.keys(): keyframe_counter[keyframe] = 0 # Add keyframe to counter if not previously added
                keyframe_counter[keyframe] += 1 # Increase count of seen map points in counter

        max = 0
        keyframe_max = None

        self.local_keyframes.clear()

        # All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
        for keyframe, counter in keyframe_counter.items():
            if counter > max:
                max = counter
                keyframe_max = keyframe
            self.local_keyframes.add(keyframe)

        # Logic to expand local window with relevant neighbouring frames can be added here
        ###

        if keyframe_max is not None:
            self.reference_keyframe = keyframe_max
            self.current_frame.reference_keyframe = self.reference_keyframe

    def update_local_points(self) -> None:
        """Update local map point variable with information from current local keyframes"""
        self.local_map_points.clear()

        for local_keyframe in self.local_keyframes:
            for map_point in local_keyframe.get_map_points():
                self.local_map_points.add(map_point)

    def update_local_map(self) -> None:
        self.update_local_keyframes()
        self.update_local_points()

    def track_local_map(self) -> bool:
        """Called when we have an estimation of camera pose and some map points are tracked in current frame.
        Retrieve local map and try to find matches to points in local map"""
        self.update_local_map()

        # Perform pose optimization
        # optimizer.pose_optimization(self.current_frame)

        self.matches_inliers = len(self.current_frame.get_map_points()) - len(self.current_frame.get_outlier_ids())
        inlier_threshold = 1
        return self.matches_inliers >= inlier_threshold


    def grab_image_monocular(self, image: cv2.Mat, timestep: int) -> Geometry.SE3:

        # Process image in necessary ways (grayscale etc)
        frame = image

        if self.state == TrackerState.NOT_INITIALIZED or self.state == TrackerState.NO_IMAGES_YET:
            self.current_frame = Frame(frame, timestep, self.init_extractor, self.camera)
        else:
            self.current_frame = Frame(frame, timestep, self.mono_extractor, self.camera)

        if self.state == TrackerState.NO_IMAGES_YET:
            self.t0 = timestep

        self.last_id = self.current_frame.id
        self.track()

        return self.current_frame.get_pose()
