
from .FeatureHandler import FeatureHandler
from .KeyframeManager import KeyframeManager

from threading import Thread
from queue import Queue
from src.common import Frame
from src.models import CameraModel
from src.measurements import CameraMeasurement
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
                keyframe_manager = KeyframeManager()
            )
            self.cameras.append(camera_instance)


        self.running = True
        self.daemon = True

    def run(self) -> None:
        while self.running:
            
            try:
                measurement: CameraMeasurement = self.incoming_queue.get_nowait()
            except Exception:
                sleep(0.1)
                continue
            
            self.handle_frame(measurement)
            sleep(0.1)

    def handle_frame(self, measurement: CameraMeasurement) -> None:
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
        

        # prev_keyframe = keyframe_manager.keyframes[-1]
        # overlapping_landmarks = FeatureHandler.match_features(measurement.frame, prev_keyframe.frame)

        keyframe_manager.add_keyframe(measurement)
        self.outgoing_queue.put(measurement)


        # print(matches[0].queryIdx, matches[0].trainIdx, matches[0].distance)
        # print(measurement.frame.descriptors[matches[0].queryIdx], local_window[-1].descriptors[matches[0].trainIdx])




    def stop(self) -> None:
        self.running = False