
from .SLAM import SLAM
from src.measurements import MeasurementType, MeasurementIdentifier
from src.measurements import VesselMeasurement, CameraMeasurement

from queue import Queue
from threading import Thread

from time import sleep

class BackendMain(Thread):
    
    def __init__(self, incoming_queue: Queue) -> None:
        super().__init__()
        self.SLAM = SLAM()
        self.incoming_queue = incoming_queue

        self.timestep = 1
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
        measurement = measurement.latest_vessel_measurement
        meas_type = MeasurementIdentifier.identify(measurement)
        match meas_type:
            case MeasurementType.VESSEL: self.vessel_measurement(measurement)
            case MeasurementType.CAMERA: pass

    def vessel_measurement(self, measurement: VesselMeasurement) -> None:
        pose = measurement.as_pose()
        pose_noise = measurement.pose_noise()
        self.SLAM.pose_measurement(self.timestep, pose, pose_noise)
        if self.timestep > 1: self.SLAM.optimize()
        self.timestep += 1


    def stop(self) -> None:
        self.running = False