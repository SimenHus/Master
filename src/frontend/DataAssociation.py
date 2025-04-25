
from src.common import Landmark, Frame, LandmarkObservation
from .FeatureHandler import FeatureHandler
from src.measurements import CameraMeasurement

class DataAssociation:
    new_landmarks: dict[int: Landmark] = {} # Dict of new landmarks to be added, need triangulation before it is added
    new_landmarks: set[Landmark] = set()
    triangulated_landmarks: set[Landmark] = set() # Dict of landmarks where initial triangulation is performed
    new_landmark_index = 1 # Index to start for new landmarks

    # def associate_local(self, frame: Frame, prev_frame: Frame) -> tuple[list[Landmark], list[Landmark]]:
    #     'Compares frames and returns tuple containing list of observed new landmarks and list of observed existing landmarks'
    #     kp1, desc1 = frame.keypoints, frame.descriptors
    #     kp2, desc2 = prev_frame.keypoints, prev_frame.descriptors

    #     for kp, desc in zip(kp1, desc1):
    #         observation = LandmarkObservation(0, keypoint)
    #         self.landmarks[self.next_landmark_id] = Landmark(self.next_landmark_id, descriptor)
    #         self.landmarks[self.next_landmark_id].observations.append(observation)
    #         self.next_landmark_id += 1
    #     if len(keyframe_manager.keyframes) <= 1: return

    def append_observation(self, observation: LandmarkObservation, landmark: Landmark, measurement: CameraMeasurement) -> None:
        print(keypoint, type(keypoint))
        camera_id = measurement.camera_id
        timestep = measurement.timestep
        landmark_id = self.new_landmark_index
        landmark_observation = LandmarkObservation(camera_id, timestep, keypoint)
        new_landmark = Landmark(landmark_id, descriptor.descriptor)
        new_landmark.observations.append(landmark_observation)

        self.descriptors.add(descriptor)
        self.descriptor_mapping[descriptor] = landmark_id
        self.new_landmarks[landmark_id] = new_landmark
        
        self.new_landmark_index += 1


    def associate_global(self, measurement: CameraMeasurement) -> list[Landmark]:
        'Checks for landmarks in new frame and returns new landmarks. New landmarks are stored in internal set'
        keypoints, descriptors = measurement.frame.features
        new_landmarks = []
        for kp, desc in zip(keypoints, descriptors):
            observation = LandmarkObservation(measurement.camera_id, measurement.timestep, kp)
            new_landmark = Landmark(-1, desc)
            if not new_landmark in self.new_landmarks:
                self.append_observation(observation, new_landmark, measurement)