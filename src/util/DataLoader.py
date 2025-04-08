
from src.common import Frame
from dataclasses import dataclass
import numpy as np

# @dataclass
# class DatasetCamera:
#     K: 'np.ndarray[4, 4]'
#     frames: list[Frame]


# @dataclass
# class Dataset:
#     cameras: list[DatasetCamera]
#     ground_truth: list


# class DatasetBaseClass:
#     folder = None
#     camera = None
#     odometry = None
#     ground_truth = None
#     pose_measurement = None


# class EuRoC1(DatasetBaseClass):
#     folder = 'C:\Users\simen\Desktop\Prog\Dataset\mav0'
#     camera = ['cam0']
#     odometry = ['imu0']


# class DataLoader:
    
#     @staticmethod
#     def load_data(dataset: DatasetBaseClass) -> tbd