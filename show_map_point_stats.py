
import json
import csv
import numpy as np
from src.util import Geometry
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt


class MapPointStats:
    def __init__(self, map_point_file) -> None:

        with open(map_point_file, 'r') as f:
            self.map_points_dict: dict = json.load(f)


    def n_obs_hist(self, ax: plt.Axes) -> None:
        obs = []
        for id, map_point in self.map_points_dict.items():
            obs.append(len(map_point['observations']))

        ax.set_title(f'Number of MapPoints: {len(self.map_points_dict)}')
        ax.set_xlabel('Number of MapPoint observations')
        ax.set_ylabel('Number of MapPoints with certain amount of observations')
        ax.hist(np.array(obs))
        ax.grid()

    def n_per_frame_hist(self, ax: plt.Axes) -> None:
        obs = {}
        for id, map_point in self.map_points_dict.items():
            for frame_id in map_point['observations'].keys():
                frame_id = str(frame_id)
                if not frame_id in obs: obs[frame_id] = 0
                obs[frame_id] += 1

        ax.set_title(f'Number of KeyFrames: {len(obs)}')
        ax.set_xlabel('KeyFrame ID')
        ax.set_ylabel('Number of MapPoints per keyframe')
        ax.bar(obs.keys(), obs.values())
        ax.grid()

    def show(self) -> None:
        fig, axs = plt.subplots(2, 2)

        self.n_obs_hist(axs[0][0])
        self.n_per_frame_hist(axs[0][1])

        plt.show()
        

if __name__ == '__main__':
    OUTPUT_FOLDER = './output'
    map_points = f'{OUTPUT_FOLDER}/MapPoints.json'
    plotter = MapPointStats(map_points)
    plotter.show()