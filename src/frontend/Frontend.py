
from .FeatureHandler import FeatureHandler
from .KeyframeManager import KeyframeManager

from threading import Thread
from queue import Queue
from src.common import Frame, Landmark, LandmarkObservation
from src.models import CameraModel
from src.measurements import MeasurementType, MeasurementIdentifier
from src.measurements import VesselMeasurement, CameraMeasurement, OdometryMeasurement
from time import sleep

from dataclasses import dataclass

@dataclass
class _CameraInstance:
    model: CameraModel
    local_window: list
    local_window_size: int
    keyframe_manager: KeyframeManager

class FrontendMain(Thread):

    def __init__(self, incoming_queue: Queue, outgoing_queue: Queue, cameras: list[CameraModel]) -> None:
        super().__init__()
        self.incoming_queue = incoming_queue
        self.outgoing_queue = outgoing_queue

        self.cameras: list[_CameraInstance] = []
        self.local_window_size = 10 # Max size of local frame window
        for camera in cameras:
            camera_instance = _CameraInstance(
                model = camera,
                local_window = [],
                local_window_size = self.local_window_size,
                keyframe_manager = KeyframeManager(0, 0)
            )
            self.cameras.append(camera_instance)

        # SHOULD MAYBE BE BACKEND???
        self.landmarks = {}
        self.next_landmark_id = 0
        self.descriptor_list = []
        self.descriptor_id_map = []

        self.running = True
        self.daemon = True

    def run(self) -> None:
        while self.running:
            
            try:
                measurement = self.incoming_queue.get_nowait()
            except Exception:
                sleep(0.1)
                continue
            
            self.handle_measurement(measurement)
            sleep(0.1)


    def handle_measurement(self, measurement) -> None:
        meas_type = MeasurementIdentifier.identify(measurement)
        match meas_type:
            # case MeasurementType.VESSEL: self.vessel_measurement(measurement)
            # case MeasurementType.ODOMETRY: self.odometry_measurement(measurement)
            case MeasurementType.CAMERA: self.camera_measurement(measurement)

    def camera_measurement(self, measurement: CameraMeasurement) -> None:
        id = measurement.camera_id
        camera_instance = self.cameras[id]
        local_window = camera_instance.local_window
        keyframe_manager = camera_instance.keyframe_manager

        keypoints, descriptors = FeatureHandler.extract_features(measurement.frame)
        measurement.frame.set_features(keypoints, descriptors)

        local_window.append(measurement.frame)
        if len(local_window) > self.local_window_size: local_window.pop(0)

        is_keyframe = keyframe_manager.determine_keyframe(measurement)
        if not is_keyframe: return

        if len(local_window) > 1:
            pos = measurement.latest_vessel_measurement.position - keyframe_manager.keyframes[-1].latest_vessel_measurement.position
            rot = measurement.latest_vessel_measurement.attitude - keyframe_manager.keyframes[-1].latest_vessel_measurement.attitude
            pos_error = (measurement.latest_vessel_measurement.pos_error + keyframe_manager.keyframes[-1].latest_vessel_measurement.pos_error)**2
            rot_error = (measurement.latest_vessel_measurement.att_error + keyframe_manager.keyframes[-1].latest_vessel_measurement.att_error)**2
            # odometry = OdometryMeasurement('', attitude=rot, position=pos, att_error=rot_error, pos_error=pos_error)
            odometry = OdometryMeasurement('', attitude=rot, position=pos)
            # self.outgoing_queue.put(odometry)

        keyframe_manager.add_keyframe(measurement)
        self.outgoing_queue.put(measurement)

        if len(keyframe_manager.keyframes) == 1:
            for keypoint, descriptor in zip(keypoints, descriptors):
                observation = LandmarkObservation(0, keypoint)
                self.landmarks[self.next_landmark_id] = Landmark(self.next_landmark_id, descriptor)
                self.landmarks[self.next_landmark_id].observations.append(observation)
                self.next_landmark_id += 1
        if len(keyframe_manager.keyframes) <= 1: return

        prev_keyframe = keyframe_manager.keyframes[-1]
        overlapping_landmarks = FeatureHandler.match_features(measurement.frame, prev_keyframe.frame)

        # print(matches[0].queryIdx, matches[0].trainIdx, matches[0].distance)
        # print(measurement.frame.descriptors[matches[0].queryIdx], local_window[-1].descriptors[matches[0].trainIdx])



    def stop(self) -> None:
        self.running = False