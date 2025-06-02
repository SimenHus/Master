

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
        self.load_gt('./datasets/osl/data')

        self.camera, self.Trc_gt = DataLoader.load_stx_camera()

    def load_map_points(self, path):
        with open(path, 'r') as f: self.map_points = json.load(f)

    def load_keyframes(self, path):
        with open(path, 'r') as f: self.keyframes = json.load(f)
        self.kf_map = {} # timestep: kf_id
        for kf_id, keyframe in self.keyframes.items():
            self.kf_map[keyframe['timestep']] = kf_id


    def load_gt(self, path):
        stx_data = DataLoader.load_stx_data()
        keyframe_timesteps = [kf['timestep'] for kf in self.keyframes.values()]

        # filtered_data = [data for data in stx_data if data.timestep in keyframe_timesteps]
        filtered_data = stx_data

        sorted_data = sorted(filtered_data, key=lambda x: x.timestep)

        self.states: dict[int: Geometry.State] = {}

        for data in sorted_data:
            self.states[data.timestep] = data.state


    def start(self):
        priors = [Geometry.SE3()]
        anchor_intervals = -1
        HA_interval = -1

        vel_sigmas = np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.1]) * 1e-1
        prior_sigmas = np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3]) * 1e-2
        extrinsic_prior_sigmas = np.append([1e-2]*3, [1e-3]*3)
        between_sigma = np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3]) * 1e-2
        HA_sigmas = np.array([0.1, 0.1, 0.1, 0.03, 0.03, 0.03]) * 1e0

        self.Trc_traj: list[Geometry.SE3] = []
        last_map_points_kf_id = None
        last_map_points = {}
        for i, (kf_id, keyframe) in enumerate(self.keyframes.items()):
            kf_id_int = int(kf_id)
            extra_optims = 0

            current_map_points = {}
            for map_point in self.map_points.values():
                observations = map_point['observations']
                if kf_id not in observations.keys(): continue # Map point not in frame
                mp_id = int(map_point['id'])
                kp = keyframe['keypoints_und'][observations[kf_id]]
                current_map_points[mp_id] = kp

            if last_map_points_kf_id is None:
                last_map_points_kf_id = kf_id_int
                last_map_points = current_map_points
                self.optimizer.add_pose_prior(priors[0], kf_id_int, NodeType.CAMERA, prior_sigmas)
                self.optimizer.add_node(priors[0], kf_id_int, NodeType.CAMERA)
                continue
            
            mp_ids = []
            mp1 = []
            mp2 = []
            for mp_id in current_map_points.keys():
                if mp_id in last_map_points:
                    mp_ids.append(mp_id)
                    mp1.append(current_map_points[mp_id])
                    mp2.append(last_map_points[mp_id])
            success, Tc1c2, triangulated_points, mask = self.camera.reconstruct_with_sorted_pixel_lists(np.array(mp1), np.array(mp2))
            if not success: continue
            
            Twc2 = self.optimizer.get_node_estimate(last_map_points_kf_id, NodeType.CAMERA).compose(Tc1c2)
            self.optimizer.add_node(Twc2, kf_id_int, NodeType.CAMERA)
            for j in range(len(mp1)):
                if mask[j]: continue
                self.optimizer.add_projection_factor(mp_ids[j], mp1[j], kf_id_int, self.camera)
                self.optimizer.add_projection_factor(mp_ids[j], mp2[j], last_map_points_kf_id, self.camera)
                if self.optimizer.get_node_estimate(mp_ids[j], NodeType.LANDMARK) is None:
                    self.optimizer.add_node(triangulated_points[j], mp_ids[j], NodeType.LANDMARK)

            try:
                self.optimizer.optimize(extra_optims) # Optimize after at least two timesteps have passed
            except Exception as e:
                print(e)
                break
            finally:
                print(i)

    def plot_gt(self, t, pos_axs: list[plt.Axes], ang_axs: list[plt.Axes]) -> None:
        cam_traj = np.empty([len(t), 6])
        last_Twr = None
        last_Twc = None
        for i, (kf_id, keyframe) in enumerate(self.keyframes.items()):
            timestep = keyframe['timestep']
            state = self.states[timestep]
            Twr = state.pose
            if last_Twr is None:
                Twc = Geometry.SE3()
            else:
                Tr1r2 = Geometry.SE3.between(last_Twr, Twr)
                Tc1c2 = self.Trc_gt.inverse() * Tr1r2 * self.Trc_gt
                Twc = last_Twc.compose(Tc1c2)

            cam_traj[i, :] = Geometry.SE3.as_vector(Twc)

            last_Twr = Twr
            last_Twc = Twc

        for i in range(len(pos_axs)):
            ang_axs[i].plot(t, cam_traj[:, i], label='Cam-gt')

            pos_axs[i].plot(t, cam_traj[:, i+3], label='Cam-gt')


    def plot_estim(self, t, pos_axs: list[plt.Axes], ang_axs: list[plt.Axes]) -> None:
        cam_traj = np.empty([len(t), 6])
        for i, (kf_id, keyframe) in enumerate(self.keyframes.items()):
            Twc = self.optimizer.get_node_estimate(int(kf_id), NodeType.CAMERA)
            if not Twc:
                cam_traj[i, :] = np.nan
                continue
            cam_traj[i, :] = Geometry.SE3.as_vector(Twc)

        for i in range(len(pos_axs)):
            ang_axs[i].plot(t, cam_traj[:, i], label='Cam-estim')

            pos_axs[i].plot(t, cam_traj[:, i+3], label='Cam-estim')


    def plot_extrinsics(self, t, pos_axs: list[plt.Axes], ang_axs: list[plt.Axes]) -> None:
        traj = np.empty([len(t), 6])
        traj_gt = np.empty([len(t), 6])
        gt = Geometry.SE3.as_vector(self.Trc_gt)
        for i, Trc in enumerate(self.Trc_traj):
            traj_gt[i, :] = gt
            if not Trc:
                traj[i, :] = np.nan
                continue
            traj[i, :] = Geometry.SE3.as_vector(Trc)

        for i in range(len(pos_axs)):
            ang_axs[i].plot(t, traj[:, i], label='Trc-traj')
            ang_axs[i].plot(t, traj_gt[:, i], label='Trc-gt')
            pos_axs[i].plot(t, traj[:, i+3], label='Trc-traj')
            pos_axs[i].plot(t, traj_gt[:, i+3], label='Trc-gt')


    def show(self) -> None:
        fig, axs = plt.subplots(2, 3)
        # _, extr_axs = plt.subplots(2, 3)
        pos_axs, ang_axs = axs[0], axs[1]
        # pos_axs_ext, ang_axs_ext = extr_axs[0], extr_axs[1]
        t = np.arange(len(self.keyframes))
        self.plot_gt(t, pos_axs, ang_axs)
        self.plot_estim(t, pos_axs, ang_axs)
        # self.plot_extrinsics(t, pos_axs_ext, ang_axs_ext)

        ang_labels = ['roll', 'pitch', 'yaw']
        pos_labels = ['x', 'y', 'z']
        for i in range(3):
            pos_axs[i].grid()
            pos_axs[i].legend()
            pos_axs[i].set_title(pos_labels[i])
            ang_axs[i].grid()
            ang_axs[i].legend()
            ang_axs[i].set_title(ang_labels[i])

            # pos_axs_ext[i].grid()
            # pos_axs_ext[i].legend()
            # pos_axs_ext[i].set_title(pos_labels[i])
            # ang_axs_ext[i].grid()
            # ang_axs_ext[i].legend()
            # ang_axs_ext[i].set_title(ang_labels[i])
            


        plt.show()
            
if __name__ == '__main__':

    app = Application()
    app.start()
    graph, estimate = app.optimizer.get_visualization_variables()
    Visualization.FactorGraphVisualization.draw_factor_graph('./output/', graph, estimate)

    app.show()
    
