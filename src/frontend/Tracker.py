
from src.structs import Frame
from src.backend import LoopClosing
from src.mapping import LocalMapping
from src.atlas import Atlas

from enum import Enum


class TrackerState(Enum):
    SYSTEM_NOT_READY = -1
    NO_IMAGES_YET = 0
    NOT_INITIALIZED = 1
    OK = 2
    # RECENTLY_LOST = 3
    # LOST = 4
    # OK_KLT = 5

# https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/src/Tracking.cc#L1794
# and https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/include/Tracking.h#L138
class Tracker:
    # Frame variables
    current_frame: Frame
    last_frame: Frame
    initial_frame: Frame
    
    t0: int

    # Pointers to other threads
    loop_closing: LoopClosing
    local_mapper: LocalMapping

    atlas: Atlas
    created_map: bool
    map_updated: bool

    state: int
    last_processed_state: int
    
    def track(self) -> None:
        current_map = self.atlas.get_current_map()
        if not current_map: print('ERROR: No active map in atlas')

        if self.state != TrackerState.NO_IMAGES_YET:
            pass
#     if(mState!=NO_IMAGES_YET)
#     {
#         if(mLastFrame.mTimeStamp>mCurrentFrame.mTimeStamp)
#         {
#             cerr << "ERROR: Frame with a timestamp older than previous frame detected!" << endl;
#             unique_lock<mutex> lock(mMutexImuQueue);
#             mlQueueImuData.clear();
#             CreateMapInAtlas();
#             return;
#         }
#         else if(mCurrentFrame.mTimeStamp>mLastFrame.mTimeStamp+1.0)
#         {
#             // cout << mCurrentFrame.mTimeStamp << ", " << mLastFrame.mTimeStamp << endl;
#             // cout << "id last: " << mLastFrame.mnId << "    id curr: " << mCurrentFrame.mnId << endl;
#             if(mpAtlas->isInertial())
#             {

#                 if(mpAtlas->isImuInitialized())
#                 {
#                     cout << "Timestamp jump detected. State set to LOST. Reseting IMU integration..." << endl;
#                     if(!pCurrentMap->GetIniertialBA2())
#                     {
#                         mpSystem->ResetActiveMap();
#                     }
#                     else
#                     {
#                         CreateMapInAtlas();
#                     }
#                 }
#                 else
#                 {
#                     cout << "Timestamp jump detected, before IMU initialization. Reseting..." << endl;
#                     mpSystem->ResetActiveMap();
#                 }
#                 return;
#             }

#         }
#     }

        if self.state == TrackerState.NO_IMAGES_YET: self.state = TrackerState.NOT_INITIALIZED

        self.last_processed_state = self.state

        self.created_map = False

        # // Get Map Mutex -> Map cannot be changed
        # unique_lock<mutex> lock(pCurrentMap->mMutexMapUpdate);

        self.map_updated = False

#     int nCurMapChangeIndex = pCurrentMap->GetMapChangeIndex();
#     int nMapChangeIndex = pCurrentMap->GetLastMapChange();
#     if(nCurMapChangeIndex>nMapChangeIndex)
#     {
#         pCurrentMap->SetLastMapChange(nCurMapChangeIndex);
#         mbMapUpdated = true;
#     }


#     if(mState==NOT_INITIALIZED)
#     {
#         if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD)
#         {
#             StereoInitialization();
#         }
#         else
#         {
#             MonocularInitialization();
#         }

#         //mpFrameDrawer->Update(this);

#         if(mState!=OK) // If rightly initialized, mState=OK
#         {
#             mLastFrame = Frame(mCurrentFrame);
#             return;
#         }

#         if(mpAtlas->GetAllMaps().size() == 1)
#         {
#             mnFirstFrameId = mCurrentFrame.mnId;
#         }
#     }
#     else
#     {
#         // System is initialized. Track Frame.
#         bool bOK;

# #ifdef REGISTER_TIMES
#         std::chrono::steady_clock::time_point time_StartPosePred = std::chrono::steady_clock::now();
# #endif

#         // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
#         if(!mbOnlyTracking)
#         {

#             // State OK
#             // Local Mapping is activated. This is the normal behaviour, unless
#             // you explicitly activate the "only tracking" mode.
#             if(mState==OK)
#             {

#                 // Local Mapping might have changed some MapPoints tracked in last frame
#                 CheckReplacedInLastFrame();

#                 if((!mbVelocity && !pCurrentMap->isImuInitialized()) || mCurrentFrame.mnId<mnLastRelocFrameId+2)
#                 {
#                     Verbose::PrintMess("TRACK: Track with respect to the reference KF ", Verbose::VERBOSITY_DEBUG);
#                     bOK = TrackReferenceKeyFrame();
#                 }
#                 else
#                 {
#                     Verbose::PrintMess("TRACK: Track with motion model", Verbose::VERBOSITY_DEBUG);
#                     bOK = TrackWithMotionModel();
#                     if(!bOK)
#                         bOK = TrackReferenceKeyFrame();
#                 }


#                 if (!bOK)
#                 {
#                     if ( mCurrentFrame.mnId<=(mnLastRelocFrameId+mnFramesToResetIMU) &&
#                          (mSensor==System::IMU_MONOCULAR || mSensor==System::IMU_STEREO || mSensor == System::IMU_RGBD))
#                     {
#                         mState = LOST;
#                     }
#                     else if(pCurrentMap->KeyFramesInMap()>10)
#                     {
#                         // cout << "KF in map: " << pCurrentMap->KeyFramesInMap() << endl;
#                         mState = RECENTLY_LOST;
#                         mTimeStampLost = mCurrentFrame.mTimeStamp;
#                     }
#                     else
#                     {
#                         mState = LOST;
#                     }
#                 }
#             }
#             else
#             {

#                 if (mState == RECENTLY_LOST)
#                 {
#                     Verbose::PrintMess("Lost for a short time", Verbose::VERBOSITY_NORMAL);

#                     bOK = true;
#                     if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD))
#                     {
#                         if(pCurrentMap->isImuInitialized())
#                             PredictStateIMU();
#                         else
#                             bOK = false;

#                         if (mCurrentFrame.mTimeStamp-mTimeStampLost>time_recently_lost)
#                         {
#                             mState = LOST;
#                             Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
#                             bOK=false;
#                         }
#                     }
#                     else
#                     {
#                         // Relocalization
#                         bOK = Relocalization();
#                         //std::cout << "mCurrentFrame.mTimeStamp:" << to_string(mCurrentFrame.mTimeStamp) << std::endl;
#                         //std::cout << "mTimeStampLost:" << to_string(mTimeStampLost) << std::endl;
#                         if(mCurrentFrame.mTimeStamp-mTimeStampLost>3.0f && !bOK)
#                         {
#                             mState = LOST;
#                             Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
#                             bOK=false;
#                         }
#                     }
#                 }
#                 else if (mState == LOST)
#                 {

#                     Verbose::PrintMess("A new map is started...", Verbose::VERBOSITY_NORMAL);

#                     if (pCurrentMap->KeyFramesInMap()<10)
#                     {
#                         mpSystem->ResetActiveMap();
#                         Verbose::PrintMess("Reseting current map...", Verbose::VERBOSITY_NORMAL);
#                     }else
#                         CreateMapInAtlas();

#                     if(mpLastKeyFrame)
#                         mpLastKeyFrame = static_cast<KeyFrame*>(NULL);

#                     Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

#                     return;
#                 }
#             }

#         }
#         else
#         {
#             // Localization Mode: Local Mapping is deactivated (TODO Not available in inertial mode)
#             if(mState==LOST)
#             {
#                 if(mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
#                     Verbose::PrintMess("IMU. State LOST", Verbose::VERBOSITY_NORMAL);
#                 bOK = Relocalization();
#             }
#             else
#             {
#                 if(!mbVO)
#                 {
#                     // In last frame we tracked enough MapPoints in the map
#                     if(mbVelocity)
#                     {
#                         bOK = TrackWithMotionModel();
#                     }
#                     else
#                     {
#                         bOK = TrackReferenceKeyFrame();
#                     }
#                 }
#                 else
#                 {
#                     // In last frame we tracked mainly "visual odometry" points.

#                     // We compute two camera poses, one from motion model and one doing relocalization.
#                     // If relocalization is sucessfull we choose that solution, otherwise we retain
#                     // the "visual odometry" solution.

#                     bool bOKMM = false;
#                     bool bOKReloc = false;
#                     vector<MapPoint*> vpMPsMM;
#                     vector<bool> vbOutMM;
#                     Sophus::SE3f TcwMM;
#                     if(mbVelocity)
#                     {
#                         bOKMM = TrackWithMotionModel();
#                         vpMPsMM = mCurrentFrame.mvpMapPoints;
#                         vbOutMM = mCurrentFrame.mvbOutlier;
#                         TcwMM = mCurrentFrame.GetPose();
#                     }
#                     bOKReloc = Relocalization();

#                     if(bOKMM && !bOKReloc)
#                     {
#                         mCurrentFrame.SetPose(TcwMM);
#                         mCurrentFrame.mvpMapPoints = vpMPsMM;
#                         mCurrentFrame.mvbOutlier = vbOutMM;

#                         if(mbVO)
#                         {
#                             for(int i =0; i<mCurrentFrame.N; i++)
#                             {
#                                 if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
#                                 {
#                                     mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
#                                 }
#                             }
#                         }
#                     }
#                     else if(bOKReloc)
#                     {
#                         mbVO = false;
#                     }

#                     bOK = bOKReloc || bOKMM;
#                 }
#             }
#         }

#         if(!mCurrentFrame.mpReferenceKF)
#             mCurrentFrame.mpReferenceKF = mpReferenceKF;

# #ifdef REGISTER_TIMES
#         std::chrono::steady_clock::time_point time_EndPosePred = std::chrono::steady_clock::now();

#         double timePosePred = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndPosePred - time_StartPosePred).count();
#         vdPosePred_ms.push_back(timePosePred);
# #endif


# #ifdef REGISTER_TIMES
#         std::chrono::steady_clock::time_point time_StartLMTrack = std::chrono::steady_clock::now();
# #endif
#         // If we have an initial estimation of the camera pose and matching. Track the local map.
#         if(!mbOnlyTracking)
#         {
#             if(bOK)
#             {
#                 bOK = TrackLocalMap();

#             }
#             if(!bOK)
#                 cout << "Fail to track local map!" << endl;
#         }
#         else
#         {
#             // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
#             // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
#             // the camera we will use the local map again.
#             if(bOK && !mbVO)
#                 bOK = TrackLocalMap();
#         }

#         if(bOK)
#             mState = OK;
#         else if (mState == OK)
#         {
#             if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
#             {
#                 Verbose::PrintMess("Track lost for less than one second...", Verbose::VERBOSITY_NORMAL);
#                 if(!pCurrentMap->isImuInitialized() || !pCurrentMap->GetIniertialBA2())
#                 {
#                     cout << "IMU is not or recently initialized. Reseting active map..." << endl;
#                     mpSystem->ResetActiveMap();
#                 }

#                 mState=RECENTLY_LOST;
#             }
#             else
#                 mState=RECENTLY_LOST; // visual to lost

#             /*if(mCurrentFrame.mnId>mnLastRelocFrameId+mMaxFrames)
#             {*/
#                 mTimeStampLost = mCurrentFrame.mTimeStamp;
#             //}
#         }

#         // Save frame if recent relocalization, since they are used for IMU reset (as we are making copy, it shluld be once mCurrFrame is completely modified)
#         if((mCurrentFrame.mnId<(mnLastRelocFrameId+mnFramesToResetIMU)) && (mCurrentFrame.mnId > mnFramesToResetIMU) &&
#            (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && pCurrentMap->isImuInitialized())
#         {
#             // TODO check this situation
#             Verbose::PrintMess("Saving pointer to frame. imu needs reset...", Verbose::VERBOSITY_NORMAL);
#             Frame* pF = new Frame(mCurrentFrame);
#             pF->mpPrevFrame = new Frame(mLastFrame);

#             // Load preintegration
#             pF->mpImuPreintegratedFrame = new IMU::Preintegrated(mCurrentFrame.mpImuPreintegratedFrame);
#         }

#         if(pCurrentMap->isImuInitialized())
#         {
#             if(bOK)
#             {
#                 if(mCurrentFrame.mnId==(mnLastRelocFrameId+mnFramesToResetIMU))
#                 {
#                     cout << "RESETING FRAME!!!" << endl;
#                     ResetFrameIMU();
#                 }
#                 else if(mCurrentFrame.mnId>(mnLastRelocFrameId+30))
#                     mLastBias = mCurrentFrame.mImuBias;
#             }
#         }

# #ifdef REGISTER_TIMES
#         std::chrono::steady_clock::time_point time_EndLMTrack = std::chrono::steady_clock::now();

#         double timeLMTrack = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndLMTrack - time_StartLMTrack).count();
#         vdLMTrack_ms.push_back(timeLMTrack);
# #endif

#         // Update drawer
#         mpFrameDrawer->Update(this);
#         if(mCurrentFrame.isSet())
#             mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose());

#         if(bOK || mState==RECENTLY_LOST)
#         {
#             // Update motion model
#             if(mLastFrame.isSet() && mCurrentFrame.isSet())
#             {
#                 Sophus::SE3f LastTwc = mLastFrame.GetPose().inverse();
#                 mVelocity = mCurrentFrame.GetPose() * LastTwc;
#                 mbVelocity = true;
#             }
#             else {
#                 mbVelocity = false;
#             }

#             if(mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
#                 mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose());

#             // Clean VO matches
#             for(int i=0; i<mCurrentFrame.N; i++)
#             {
#                 MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
#                 if(pMP)
#                     if(pMP->Observations()<1)
#                     {
#                         mCurrentFrame.mvbOutlier[i] = false;
#                         mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
#                     }
#             }

#             // Delete temporal MapPoints
#             for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
#             {
#                 MapPoint* pMP = *lit;
#                 delete pMP;
#             }
#             mlpTemporalPoints.clear();

# #ifdef REGISTER_TIMES
#             std::chrono::steady_clock::time_point time_StartNewKF = std::chrono::steady_clock::now();
# #endif
#             bool bNeedKF = NeedNewKeyFrame();

#             // Check if we need to insert a new keyframe
#             // if(bNeedKF && bOK)
#             if(bNeedKF && (bOK || (mInsertKFsLost && mState==RECENTLY_LOST &&
#                                    (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD))))
#                 CreateNewKeyFrame();

# #ifdef REGISTER_TIMES
#             std::chrono::steady_clock::time_point time_EndNewKF = std::chrono::steady_clock::now();

#             double timeNewKF = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndNewKF - time_StartNewKF).count();
#             vdNewKF_ms.push_back(timeNewKF);
# #endif

#             // We allow points with high innovation (considererd outliers by the Huber Function)
#             // pass to the new keyframe, so that bundle adjustment will finally decide
#             // if they are outliers or not. We don't want next frame to estimate its position
#             // with those points so we discard them in the frame. Only has effect if lastframe is tracked
#             for(int i=0; i<mCurrentFrame.N;i++)
#             {
#                 if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
#                     mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
#             }
#         }

#         // Reset if the camera get lost soon after initialization
#         if(mState==LOST)
#         {
#             if(pCurrentMap->KeyFramesInMap()<=10)
#             {
#                 mpSystem->ResetActiveMap();
#                 return;
#             }
#             if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
#                 if (!pCurrentMap->isImuInitialized())
#                 {
#                     Verbose::PrintMess("Track lost before IMU initialisation, reseting...", Verbose::VERBOSITY_QUIET);
#                     mpSystem->ResetActiveMap();
#                     return;
#                 }

#             CreateMapInAtlas();

#             return;
#         }

#         if(!mCurrentFrame.mpReferenceKF)
#             mCurrentFrame.mpReferenceKF = mpReferenceKF;

#         mLastFrame = Frame(mCurrentFrame);
#     }




#     if(mState==OK || mState==RECENTLY_LOST)
#     {
#         // Store frame pose information to retrieve the complete camera trajectory afterwards.
#         if(mCurrentFrame.isSet())
#         {
#             Sophus::SE3f Tcr_ = mCurrentFrame.GetPose() * mCurrentFrame.mpReferenceKF->GetPoseInverse();
#             mlRelativeFramePoses.push_back(Tcr_);
#             mlpReferences.push_back(mCurrentFrame.mpReferenceKF);
#             mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
#             mlbLost.push_back(mState==LOST);
#         }
#         else
#         {
#             // This can happen if tracking is lost
#             mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
#             mlpReferences.push_back(mlpReferences.back());
#             mlFrameTimes.push_back(mlFrameTimes.back());
#             mlbLost.push_back(mState==LOST);
#         }

#     }

# #ifdef REGISTER_LOOP
#     if (Stop()) {

#         // Safe area to stop
#         while(isStopped())
#         {
#             usleep(3000);
#         }
#     }
# #endif
# }