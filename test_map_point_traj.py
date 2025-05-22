

from src.backend import Optimizer
from src.structs import Camera
from src.util import Geometry, DataLoader, Time
import json
import numpy as np

from src.visualization import Visualization

import matplotlib.pyplot as plt

class Application:
    
    def __init__(self) -> None:
        self.optimizer = Optimizer()
        self.load_map_points('./output/MapPoints.json')
        self.load_keyframes('./output/KeyFrames.json')
        self.load_ref_pose_gt('./datasets/osl/data')

        self.camera, self.Twc_gt = DataLoader.load_stx_camera('./datasets/osl')

    def load_map_points(self, path):
        with open(path, 'r') as f: self.map_points = json.load(f)

    def load_keyframes(self, path):
        # with open(path, 'r') as f: self.keyframes = sorted(json.load(f), key=lambda x: x['id'])
        with open(path, 'r') as f: self.keyframes = json.load(f)

    def load_ref_pose_gt(self, path):
        poses = DataLoader.load_stx_poses(path)
        pose_dict = {pose.timestep: pose.pose for pose in poses}
        self.poses = pose_dict


    def start(self):
        for kf_id, keyframe in self.keyframes.items():
            id0 = int(kf_id)
            break

        # T0 = self.poses[t0]
        # Twc0 = T0.compose(self.Twc_gt)
        Twc0 = Geometry.SE3()
        prior_sigmas = [0.5, 0.5, 0.5, 1, 1, 1]
        self.optimizer.add_camera_prior(Twc0, id0, prior_sigmas) # Add prior for first camera pose based on initial guess of extrinsics
        for i, (kf_id, keyframe) in enumerate(self.keyframes.items()):
            # ref_pose = self.poses[keyframe['timestep']] # Get pose from file
            # cam_pose = ref_pose.compose(self.Twc_gt)
            Tc0c = Geometry.SE3(keyframe['Twc'])
            cam_pose = Twc0.compose(Tc0c)
            kf_id_int = int(keyframe['id'])
            self.optimizer.add_camera_node(cam_pose, kf_id_int) # Add node for camera pose

            for map_point in self.map_points.values():
                observations = map_point['observations']
                if len(observations) < 6: continue # Skip map points with few observations
                if kf_id not in observations.keys(): continue # Map point not in frame
                mp_id = int(map_point['id'])
                kp = keyframe['keypoints_und'][observations[kf_id]]
                self.optimizer.update_projection_factor(mp_id, kp, kf_id_int, self.camera)
            if i > 0:
                self.optimizer.optimize() # Optimize after at least two timesteps have passed
                if i % 2 == 0: 
                    self.optimizer.batch_optimize()
            # print(i)
            if i == 3: break

    def plot_gt(self, t, pos_axs: list[plt.Axes], ang_axs: list[plt.Axes]) -> None:
        ref_traj = np.empty([len(self.keyframes), 6])
        cam_traj = np.empty([len(self.keyframes), 6])
        for i, (kf_id, keyframe) in enumerate(self.keyframes.items()):
            ref_pose = self.poses[keyframe['timestep']]
            cam_pose = ref_pose.compose(self.Twc_gt)
            ref_traj[i, :] = Geometry.SE3.to_STX(ref_pose)
            cam_traj[i, :] = Geometry.SE3.to_STX(cam_pose)
        ang_labels = ['roll', 'pitch', 'yaw']
        pos_labels = ['x', 'y', 'z']
        for i, (ang_label, pos_label) in enumerate(zip(ang_labels, pos_labels)):
            ang_axs[i].set_title(ang_label)
            ang_axs[i].plot(t, ref_traj[:, i], label='Ref-gt')
            ang_axs[i].plot(t, cam_traj[:, i], label='Cam-gt')

            pos_axs[i].set_title(pos_label)
            pos_axs[i].plot(t, ref_traj[:, i+3], label='Ref-gt')
            pos_axs[i].plot(t, cam_traj[:, i+3], label='Cam-gt')

        for ax1, ax2 in zip(pos_axs, ang_axs):
            ax1.grid()
            ax2.grid()
            ax1.legend()
            ax2.legend()



    def show(self) -> None:
        fig, axs = plt.subplots(2, 3)

        t = np.arange(len(self.keyframes))
        self.plot_gt(t, axs[0], axs[1])


        plt.show()
            
if __name__ == '__main__':

    app = Application()
    # app.start()
    app.show()
    # graph, estimate = app.optimizer.get_visualization_variables()
    # Visualization.FactorGraphVisualization.draw_factor_graph('./output/', graph, estimate)
