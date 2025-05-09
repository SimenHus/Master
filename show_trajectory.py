
import json
import numpy as np
from src.util import Geometry
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt


class PlotTrajectory:

    def __init__(self, keyframes_path) -> None:
        
        with open(keyframes_path, 'r') as f:
            self.keyframes_dict: dict = json.load(f)


    def plot(self) -> None:
        
        timesteps = []
        x_traj = []
        for id, kf in self.keyframes_dict.items():
            pose = Geometry.SE3(kf['Twc'])
            timesteps.append(kf['timestep'])
            x_traj.append(pose.x())

        fig, ax = plt.subplots()

        ax.plot(timesteps, x_traj, label='x')
        ax.grid()
        ax.legend()
        plt.show()

        


if __name__ == '__main__':
    OUTPUT_FOLDER = './output'
    keyframes = f'{OUTPUT_FOLDER}/KeyFrames.json'
    plotter = PlotTrajectory(keyframes)

    plotter.plot()