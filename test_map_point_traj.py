

from src.backend import Optimizer
from src.structs import Camera
from src.util import Geometry, DataLoader, Time, Geode
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
        with open(path, 'r') as f: self.keyframes = json.load(f)

    def load_ref_pose_gt(self, path):
        stx_data = DataLoader.load_stx_data(path)
        keyframe_timesteps = [kf['timestep'] for kf in self.keyframes.values()]
        filtered_data = [data for data in stx_data if data.timestep in keyframe_timesteps]

        sorted_data = sorted(filtered_data, key=lambda x: x.timestep)
        ref_lla = sorted_data[0].lla
        self.poses: dict[int: Geometry.SE3] = {}
        for data in sorted_data:
            NED = Geode.Transformation.LLA_to_NED(data.lla, ref_lla)
            vals = np.append(data.att, NED)
            self.poses[data.timestep] = Geometry.SE3.from_STX(vals)


    def start(self):
        init_ids = []
        init_ts = []
        for i, (kf_id, keyframe) in enumerate(self.keyframes.items()):
            if i > 1: break
            init_ids.append(int(kf_id))
            init_ts.append(keyframe['timestep'])

        T0 = self.poses[init_ts[0]]
        T1 = self.poses[init_ts[1]]
        Twc0 = T0.compose(self.Twc_gt)
        Twc1 = T1.compose(self.Twc_gt)
        # Twc0 = Geometry.SE3()
        prior_sigmas = [0.1, 0.1, 0.1, 0.3, 0.3, 0.3]
        self.optimizer.add_camera_prior(Twc0, init_ids[0], prior_sigmas) # Add prior for first camera pose based on initial guess of extrinsics
        self.optimizer.add_camera_prior(Twc1, init_ids[1], prior_sigmas) # Add prior for first camera pose based on initial guess of extrinsics
        for i, (kf_id, keyframe) in enumerate(self.keyframes.items()):
            kf_id_int = int(keyframe['id'])

            ref_pose = self.poses[keyframe['timestep']] # Get ref pose gt from file
            Tc0c = Geometry.SE3(keyframe['Twc']) # Get estimated pose from camera triangulation

            # cam_pose = Twc0.compose(Tc0c)
            cam_pose = ref_pose.compose(self.Twc_gt)

            self.optimizer.add_camera_node(cam_pose, kf_id_int) # Add node for camera pose

            for map_point in self.map_points.values():
                observations = map_point['observations']
                if len(observations) < 3: continue # Skip map points with few observations
                if kf_id not in observations.keys(): continue # Map point not in frame
                mp_id = int(map_point['id'])
                kp = keyframe['keypoints_und'][observations[kf_id]]
                self.optimizer.update_projection_factor(mp_id, kp, kf_id_int, self.camera)
            if i > 0:
                self.optimizer.optimize() # Optimize after at least two timesteps have passed
            print(i)
            # if i == 9: break


    def plot_gt(self, t, pos_axs: list[plt.Axes], ang_axs: list[plt.Axes]) -> None:
        ref_traj = np.empty([len(self.keyframes), 6])
        cam_traj = np.empty([len(self.keyframes), 6])
        for i, (kf_id, keyframe) in enumerate(self.keyframes.items()):
            # print(kf_id, Time.TimeConversion.POSIX_to_STX(keyframe['timestep']))
            ref_pose = self.poses[keyframe['timestep']]
            cam_pose = ref_pose.compose(self.Twc_gt)
            ref_traj[i, :] = Geometry.SE3.to_STX(ref_pose)
            cam_traj[i, :] = Geometry.SE3.to_STX(cam_pose)

        for i in range(len(pos_axs)):
            ang_axs[i].plot(t, ref_traj[:, i], label='Ref-gt')
            ang_axs[i].plot(t, cam_traj[:, i], label='Cam-gt')

            pos_axs[i].plot(t, ref_traj[:, i+3], label='Ref-gt')
            pos_axs[i].plot(t, cam_traj[:, i+3], label='Cam-gt')


    def plot_estim(self, t, pos_axs: list[plt.Axes], ang_axs: list[plt.Axes]) -> None:
        # ref_traj = np.empty([len(self.keyframes), 6])
        cam_traj = np.empty([len(self.keyframes), 6])
        for i, (kf_id, keyframe) in enumerate(self.keyframes.items()):
            kf_id_int = int(kf_id)
            # ref_pose = self.poses[keyframe['timestep']]
            cam_pose = self.optimizer.get_camera_node_estimate(kf_id_int)
            if not cam_pose:
                cam_traj[i, :] = np.nan
                continue
            # ref_traj[i, :] = Geometry.SE3.to_STX(ref_pose)
            cam_traj[i, :] = Geometry.SE3.to_STX(cam_pose)

        for i in range(len(pos_axs)):
            # ang_axs[i].plot(t, ref_traj[:, i], label='Ref-estim')
            ang_axs[i].plot(t, cam_traj[:, i], label='Cam-estim')

            # pos_axs[i].plot(t, ref_traj[:, i+3], label='Ref-estim')
            pos_axs[i].plot(t, cam_traj[:, i+3], label='Cam-estim')


    def show(self) -> None:
        fig, axs = plt.subplots(2, 3)
        pos_axs, ang_axs = axs[0], axs[1]
        t = np.arange(len(self.keyframes))
        self.plot_gt(t, pos_axs, ang_axs)
        self.plot_estim(t, pos_axs, ang_axs)

        ang_labels = ['roll', 'pitch', 'yaw']
        pos_labels = ['x', 'y', 'z']
        for i in range(3):
            pos_axs[i].grid()
            pos_axs[i].legend()
            pos_axs[i].set_title(pos_labels[i])
            ang_axs[i].grid()
            ang_axs[i].legend()
            ang_axs[i].set_title(ang_labels[i])


        plt.show()
            
if __name__ == '__main__':

    app = Application()
    app.start()
    app.show()
    graph, estimate = app.optimizer.get_visualization_variables()
    Visualization.FactorGraphVisualization.draw_factor_graph('./output/', graph, estimate)
