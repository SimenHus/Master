
import json
import csv
import numpy as np
from src.util import Geometry
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt


class PlotTrajectory:

    def __init__(self, keyframes_path, ground_truth_file) -> None:
        
        with open(keyframes_path, 'r') as f:
            self.keyframes_dict: dict = json.load(f)

        ground_truth = {} # Timestep: [states]
        with open(ground_truth_file, newline='') as csvf:
            reader = csv.DictReader(csvf)
            for row in reader:
                translation = [row[' p_RS_R_x [m]'], row[' p_RS_R_y [m]'], row[' p_RS_R_z [m]']]
                ground_truth[int(row['#timestamp'])] = translation
        self.ground_truth = ground_truth

        self.Trw = Geometry.SE3(np.array([
            [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
            [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
            [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
            [0.0, 0.0, 0.0, 1.0]
        ]))


    def translation(self) -> 'np.ndarray[3, n]':
        traj = []
        for id, kf in self.keyframes_dict.items():
            Twc = Geometry.SE3(kf['Twc'])
            Trc = self.Trw.compose(Twc)
            traj.append(Trc.translation())
        return np.array(traj).T

    def gt_translation(self) -> 'np.ndarray[3, n]':
        traj = []
        last_id = 0
        for id, kf in self.keyframes_dict.items():
            kf_timestep = int(kf['timestep'])
            gt_timesteps = list(self.ground_truth.keys())
            for i, timestep in enumerate(gt_timesteps[last_id:]):
                print(kf_timestep - timestep)
                exit()
                if kf_timestep >= timestep:
                    last_id += i
                    trans = self.ground_truth[last_id]
            traj.append(trans)

        return np.array(traj).T

    def plot(self) -> None:
        
        
        traj = self.translation()
        # gt = self.gt_translation()
        x_axis = np.arange(0, traj.shape[1], 1)
        fig, axs = plt.subplots(1, 3)

        labels = ['x', 'y', 'z']
        for i, ax in enumerate(axs):
            ax.plot(x_axis, traj[i, :], label=labels[i])
            ax.grid()
            ax.legend()
        plt.show()

        


if __name__ == '__main__':
    OUTPUT_FOLDER = './output'
    keyframes = f'{OUTPUT_FOLDER}/KeyFrames.json'
    ground_truth = '/mnt/c/Users/simen/Desktop/Prog/Dataset/mav0/state_groundtruth_estimate0/data.csv'
    plotter = PlotTrajectory(keyframes, ground_truth)

    plotter.plot()