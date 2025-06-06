
from .SLAM import SLAM
from src.measurements import MeasurementType, MeasurementIdentifier
from src.measurements import VesselMeasurement, CameraMeasurement, OdometryMeasurement

from queue import Queue
from threading import Thread

from time import sleep

class LoopClosing(Thread):
    
    def __init__(self, incoming_queue: Queue) -> None:
        super().__init__()
        self.SLAM = SLAM()
        self.incoming_queue = incoming_queue

        self.timestep = 1
        self.running = True
        self.daemon = True


        self.pose_map = {}


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
            case MeasurementType.VESSEL: self.vessel_measurement(measurement)
            case MeasurementType.ODOMETRY: self.odometry_measurement(measurement)
            case MeasurementType.CAMERA: self.camera_measurement(measurement)

    def camera_measurement(self, measurement: CameraMeasurement) -> None:
        camera_model = measurement.camera_model
        pose_id = self.pose_map[measurement.timestep]

        for landmark in measurement.landmarks:
            pixels = landmark.observations[0].keypoint.pt
            self.SLAM.landmark_measurement(pose_id, landmark.id, landmark.position, pixels, camera_model)

    def vessel_measurement(self, measurement: VesselMeasurement) -> None:
        pose = measurement.as_pose()
        pose_noise = measurement.pose_noise()
        self.SLAM.pose_measurement(self.timestep, pose, pose_noise)
        if self.timestep > 1: self.SLAM.optimize()
        self.timestep += 1

        self.pose_map[measurement.timestep] = self.timestep - 1

    def odometry_measurement(self, measurement: OdometryMeasurement) -> None:
        pose = measurement.as_pose()
        pose_noise = measurement.pose_noise()
        self.SLAM.odometry_measurement(self.timestep-1, self.timestep, pose, pose_noise)
        # if self.timestep > 1: self.SLAM.optimize()


    def stop(self) -> None:
        self.running = False