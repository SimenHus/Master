
from .KeyframeManager import KeyframeManager
from .DataAssociation import DataAssociation

from threading import Thread
from queue import Queue
from src.common import Landmark, LandmarkObservation
from src.camera import CameraModel, Frame
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
    data_assoc = DataAssociation()

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

        measurement.frame.extract_features()

        local_window.append(measurement.frame)
        if len(local_window) > self.local_window_size: local_window.pop(0)

        is_keyframe = keyframe_manager.determine_keyframe(measurement)
        if not is_keyframe: return

        if len(local_window) > 1: # Simulate odometry measurement
            pos = measurement.latest_vessel_measurement.position - keyframe_manager.keyframes[-1].latest_vessel_measurement.position
            rot = measurement.latest_vessel_measurement.attitude - keyframe_manager.keyframes[-1].latest_vessel_measurement.attitude
            pos_error = (measurement.latest_vessel_measurement.pos_error + keyframe_manager.keyframes[-1].latest_vessel_measurement.pos_error)**2
            rot_error = (measurement.latest_vessel_measurement.att_error + keyframe_manager.keyframes[-1].latest_vessel_measurement.att_error)**2
            # odometry = OdometryMeasurement('', attitude=rot, position=pos, att_error=rot_error, pos_error=pos_error)
            odometry = OdometryMeasurement('', attitude=rot, position=pos)
            # self.outgoing_queue.put(odometry)

        detected_landmarks = self.data_assoc.associate_global(measurement)
        measurement.landmarks = detected_landmarks

        measurement.camera_model = self.cameras[0].model
        keyframe_manager.add_keyframe(measurement)
        self.outgoing_queue.put(measurement.latest_vessel_measurement)
        self.outgoing_queue.put(measurement)


    def stop(self) -> None:
        self.running = False