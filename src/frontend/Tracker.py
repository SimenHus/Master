
from src.structs import Frame, KeyFrame, Camera, MapPoint
# from src.backend import LoopClosing
# from src.mapping import LocalMapping
from src.atlas import Atlas
from src.util import Geometry, DataAssociation, Logging

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
    logger = Logging.get_logger('Tracker')
    current_frame: Frame
    last_frame: Frame

    first_frame_id: int
    initial_frame_id: int
    last_init_frame_id: int
    
    t0: int

    # Pointers to other threads
    # loop_closing: LoopClosing
    # local_mapper: LocalMapping

    created_map: bool
    map_updated: bool

    last_processed_state: int


    # Monocular initialization variables
    init_last_matches: list[int]
    init_matches: list[cv2.DMatch]
    prev_matched: list[Geometry.Point2]
    init_P3D: list[Geometry.Point3]
    initial_frame: Frame
    ready_to_init: bool = False
    set_init: bool

    def __init__(self) -> None:
        self.init_id, self.last_id = 0, 0
        self.state = TrackerState.NO_IMAGES_YET
        self.atlas = Atlas()
        self.camera = Camera([458.654, 457.296, 367.215, 248.375], [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]) # More logic around camera should be added
    
    def track(self) -> None:
        current_map = self.atlas.get_current_map()
        if not current_map:
            self.logger.error('No active map found in atlas')

        if self.state != TrackerState.NO_IMAGES_YET:
            if self.last_frame.timestep > self.current_frame.timestep: # Sanity check for timesteps
                self.logger.error('Frame with timestep older than previous frame detected')
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

        if not self.current_frame.reference_keyframe: # Check if reference keyframe exists for current frame
            self.current_frame.reference_keyframe = self.reference_keyframe # If no reference keyframe found, set tracker keyframe as reference

        ok = self.track_local_map()

        if ok: self.state = TrackerState.OK

        need_KF = self.need_new_keyframe()

        if need_KF and ok:
            self.create_new_keyframe()

        if not self.current_frame.reference_keyframe: # Check if reference keyframe exists for current frame
            self.current_frame.reference_keyframe = self.reference_keyframe # If no reference keyframe found, set tracker keyframe as reference

        self.last_frame = Frame.copy(self.current_frame)



    def monocular_initialization(self) -> None:
        
        self.logger.info('Attempting to initialize monocular tracking')
        n_features_init_threshold = 100
        # Check if we have enough keypoints to initialize
        if len(self.current_frame.keypoints) < n_features_init_threshold:
            self.logger.error(f'Not enough keypoints to initialize: {len(self.current_frame.keypoints)}/{n_features_init_threshold}')
            return

        if not self.ready_to_init:
            self.initial_frame = Frame.copy(self.current_frame)
            self.last_frame = Frame.copy(self.current_frame)
            self.prev_matched = [key_und.pt for key_und in self.current_frame.keypoints_und]

            self.ready_to_init = True
            self.logger.info('Monocular initialization ready')
            return
        
        # We are ready to initialize if we reach this point
        matcher = DataAssociation.Matcher(0.9, True)
        self.init_matches = matcher.search_for_initialization(self.initial_frame, self.current_frame, self.prev_matched)
        
        n_matches = 4
        if len(self.init_matches) < n_matches: # Not enough matches
            self.ready_to_init = False
            self.logger.error(f'Monocular initialization failed, not enough matches: {len(self.init_matches)}/{n_matches}')
            return
        
        # Attempt to reconstruct a scenario using two images. This will be the initial map
        success, Tcw, self.init_P3D = self.camera.reconstruct_with_two_views(self.initial_frame.keypoints_und, self.current_frame.keypoints_und, self.init_matches)
        if not success: return
        
        self.logger.info('Monocular initialization successfull')
        self.initial_frame.set_pose(Geometry.SE3()) # Initialize the first point at the origin
        self.current_frame.set_pose(Tcw) # Initialize first movement to be the second frame and the estimated movement
        self.create_initial_map_monocular()


    def create_initial_map_monocular(self) -> None:
        # self.state = TrackerState.OK
        self.logger.info('Creating initial map for monocular tracking')
        initial_keyframe = KeyFrame(self.initial_frame, self.atlas.get_current_map())
        current_keyframe = KeyFrame(self.current_frame, self.atlas.get_current_map())

        # initial_keyframe.compute_BoW()
        # current_keyframe.compute_BoW()

        self.atlas.add_keyframe(initial_keyframe)
        self.atlas.add_keyframe(current_keyframe)

        for i, (match, P3D) in enumerate(zip(self.init_matches, self.init_P3D)):
            # Create MapPoint

            map_point = MapPoint(P3D, current_keyframe, self.atlas.get_current_map())
            initial_keyframe.add_map_point(map_point, match.queryIdx)
            current_keyframe.add_map_point(map_point, match.trainIdx)

            map_point.add_observation(initial_keyframe, match.queryIdx)
            map_point.add_observation(current_keyframe, match.trainIdx)

            map_point.compute_distinct_descriptors()
            map_point.update_normal_and_depth()

            self.current_frame.map_points[match.trainIdx] = map_point
            self.current_frame.outlier[match.trainIdx] = False

            self.atlas.add_map_point(map_point)


#     // Update Connections
#     pKFini->UpdateConnections();
#     pKFcur->UpdateConnections();

#     std::set<MapPoint*> sMPs;
#     sMPs = pKFini->GetMapPoints();

#     // Bundle Adjustment
#     Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);
#     Optimizer::GlobalBundleAdjustemnt(mpAtlas->GetCurrentMap(),20);

#     float medianDepth = pKFini->ComputeSceneMedianDepth(2);
#     float invMedianDepth;
#     if(mSensor == System::IMU_MONOCULAR)
#         invMedianDepth = 4.0f/medianDepth; // 4.0f
#     else
#         invMedianDepth = 1.0f/medianDepth;

#     if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<50) // TODO Check, originally 100 tracks
#     {
#         Verbose::PrintMess("Wrong initialization, reseting...", Verbose::VERBOSITY_QUIET);
#         mpSystem->ResetActiveMap();
#         return;
#     }

#     // Scale initial baseline
#     Sophus::SE3f Tc2w = pKFcur->GetPose();
#     Tc2w.translation() *= invMedianDepth;
#     pKFcur->SetPose(Tc2w);

#     // Scale points
#     vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
#     for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
#     {
#         if(vpAllMapPoints[iMP])
#         {
#             MapPoint* pMP = vpAllMapPoints[iMP];
#             pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
#             pMP->UpdateNormalAndDepth();
#         }
#     }

#     if (mSensor == System::IMU_MONOCULAR)
#     {
#         pKFcur->mPrevKF = pKFini;
#         pKFini->mNextKF = pKFcur;
#         pKFcur->mpImuPreintegrated = mpImuPreintegratedFromLastKF;

#         mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKFcur->mpImuPreintegrated->GetUpdatedBias(),pKFcur->mImuCalib);
#     }


#     mpLocalMapper->InsertKeyFrame(pKFini);
#     mpLocalMapper->InsertKeyFrame(pKFcur);
#     mpLocalMapper->mFirstTs=pKFcur->mTimeStamp;

#     mCurrentFrame.SetPose(pKFcur->GetPose());
#     mnLastKeyFrameId=mCurrentFrame.mnId;
#     mpLastKeyFrame = pKFcur;
#     //mnLastRelocFrameId = mInitialFrame.mnId;

#     mvpLocalKeyFrames.push_back(pKFcur);
#     mvpLocalKeyFrames.push_back(pKFini);
#     mvpLocalMapPoints=mpAtlas->GetAllMapPoints();
#     mpReferenceKF = pKFcur;
#     mCurrentFrame.mpReferenceKF = pKFcur;

#     // Compute here initial velocity
#     vector<KeyFrame*> vKFs = mpAtlas->GetAllKeyFrames();

#     Sophus::SE3f deltaT = vKFs.back()->GetPose() * vKFs.front()->GetPoseInverse();
#     mbVelocity = false;
#     Eigen::Vector3f phi = deltaT.so3().log();

#     double aux = (mCurrentFrame.mTimeStamp-mLastFrame.mTimeStamp)/(mCurrentFrame.mTimeStamp-mInitialFrame.mTimeStamp);
#     phi *= aux;

#     mLastFrame = Frame(mCurrentFrame);

#     mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

#     mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

#     mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

#     mState=OK;

#     initID = pKFcur->mnId;
# }

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
        pass # New keyframe logic

    def need_new_keyframe(self) -> bool:
        return False

    def track_local_map(self) -> bool:
        return False

    def track_reference_keyframe(self) -> bool:
        return False

    def grab_image_monocular(self, image: cv2.Mat, timestep: int) -> Geometry.SE3:

        # Process image in necessary ways (grayscale etc)
        frame = image

        # Maybe perform some checks if system is initialized?
        # if self.state == TrackerState.NOT_INITIALIZED or self.state == TrackerState.NO_IMAGES_YET:
        self.current_frame = Frame(frame, timestep, self.camera)

        if self.state == TrackerState.NO_IMAGES_YET:
            self.t0 = timestep

        self.last_id = self.current_frame.id
        self.track()

        return self.current_frame.get_pose()
