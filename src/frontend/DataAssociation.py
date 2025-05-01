
from src.common import Landmark, LandmarkObservation
from src.measurements import CameraMeasurement
from src.camera import Frame

import numpy as np

class DataAssociation:
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


    def update_triangulated_landmarks(self, landmark: Landmark) -> Landmark:
        existing_instance = next((item for item in self.triangulated_landmarks if item == landmark), None)
        self.triangulated_landmarks.remove(existing_instance)

        existing_instance.add_observation(landmark.observations)
        self.triangulated_landmarks.add(existing_instance)

        landmark.id = existing_instance.id
        return landmark

    def triangulate_landmark(self, landmark: Landmark) -> Landmark:
        existing_instance = next((item for item in self.new_landmarks if item == landmark), None)
        self.new_landmarks.remove(existing_instance)
        existing_instance.add_observation(landmark.observations)

        existing_instance.position = np.array([0, 0, 0])
        landmark.position = np.array([0, 0, 0])
        self.triangulated_landmarks.add(existing_instance)

        landmark.id = existing_instance.id
        return landmark


    def add_new_landmark(self, landmark: Landmark) -> None:
        landmark.id = self.new_landmark_index
        self.new_landmarks.add(landmark)
        self.new_landmark_index += 1


    def associate_global(self, measurement: CameraMeasurement) -> list[Landmark]:
        'Checks for landmarks in new frame and returns two lists of newly triangulated landmarks and detected existing landmarks'
        detected_landmarks = []
        for feature in measurement.frame.features:
            observation = LandmarkObservation(measurement.camera_id, measurement.timestep, feature.keypoint)
            new_landmark = Landmark(-1, feature.descriptor)
            new_landmark.add_observation(observation)
            if new_landmark in self.new_landmarks: # Check if landmark has been seen twice
                new_landmark = self.triangulate_landmark(new_landmark)
                detected_landmarks.append(new_landmark)
                continue

            if new_landmark in self.triangulated_landmarks: # Check if landmark has been seen more than twice
                new_landmark = self.update_triangulated_landmarks(new_landmark)
                detected_landmarks.append(new_landmark)
                continue

            self.add_new_landmark(new_landmark)
        return detected_landmarks