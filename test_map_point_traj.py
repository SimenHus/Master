

from src.backend import Optimizer, NodeType
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

        self.camera, self.Trc_gt = DataLoader.load_stx_camera('./datasets/osl')

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
        iter_before_optim = 1
        n_priors = 2
        prior_sigmas = np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3]) * 1e0
        anchor_sigmas = np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3]) * 1e0
        detected_mps = set()

        Trc = self.Trc_gt
        # Trc = Geometry.SE3()
        self.optimizer.add_pose_node(Trc, self.camera.id, NodeType.EXTRINSIC)
        # self.optimizer.add_pose_equality(Trc, self.camera.id, NodeType.EXTRINSIC)
        self.optimizer.add_pose_prior(Trc, self.camera.id, NodeType.EXTRINSIC, [1e-6]*6)
        for i, (kf_id, keyframe) in enumerate(self.keyframes.items()):
            kf_id_int = int(keyframe['id'])

            ref_pose = self.poses[keyframe['timestep']] # Get ref pose gt from file
            cam_pose = ref_pose.compose(Trc)
            self.optimizer.add_pose_node(cam_pose, kf_id_int, NodeType.CAMERA) # Add node for camera poses
            self.optimizer.add_reference_anchor(self.camera.id, kf_id_int, ref_pose, anchor_sigmas)

            overlap = False
            for map_point in self.map_points.values():
                observations = map_point['observations']
                if len(observations) < 3: continue # Skip map points with few observations
                if kf_id not in observations.keys(): continue # Map point not in frame
                mp_id = int(map_point['id'])
                kp = keyframe['keypoints_und'][observations[kf_id]]
                self.optimizer.update_projection_factor(mp_id, kp, kf_id_int, self.camera)
                if mp_id in detected_mps: overlap = True
                else: detected_mps.add(mp_id)

            if i < n_priors or not overlap:
                print(f'Frame - Overlap: {i}-{overlap}')
                self.optimizer.add_pose_prior(cam_pose, kf_id_int, NodeType.CAMERA, prior_sigmas) # Add prior

            if i >= iter_before_optim: self.optimizer.optimize() # Optimize after at least two timesteps have passed
            print(i)
            if i == 9: break


    def plot_gt(self, t, pos_axs: list[plt.Axes], ang_axs: list[plt.Axes]) -> None:
        ref_traj = np.empty([len(self.keyframes), 6])
        cam_traj = np.empty([len(self.keyframes), 6])
        for i, (kf_id, keyframe) in enumerate(self.keyframes.items()):
            # print(kf_id, Time.TimeConversion.POSIX_to_STX(keyframe['timestep']))
            ref_pose = self.poses[keyframe['timestep']]
            cam_pose = ref_pose.compose(self.Trc_gt)
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
            # ref_pose = self.optimizer.get_ref_node_estimate(kf_id_int)
            cam_pose = self.optimizer.get_pose_node_estimate(kf_id_int, NodeType.CAMERA)
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
    print(app.optimizer.get_pose_node_estimate(app.camera.id, NodeType.EXTRINSIC))
    
