

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
        n_priors = 2
        anchor_intervals = 5
        HA_interval = -1
        use_mps = True

        prior_sigmas = np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3]) * 1e-2
        extrinsic_prior_sigmas = np.append([1e-1]*3, [1e-1]*3)
        between_sigma = np.array([0.1, 0.1, 0.1, 0.03, 0.03, 0.03]) * 1e-2
        HA_sigmas = np.array([0.1, 0.1, 0.1, 0.03, 0.03, 0.03]) * 1e0

        noisy_vals = np.append([-1, 2, -4], [0.1, -0.1, 0.1]) * 1
        Trc_init = Geometry.SE3.as_vector(self.Trc_gt) + noisy_vals
        Trc_init = Geometry.SE3.from_vector(Trc_init, radians=False)

        self.optimizer.add_node(Trc_init, self.camera.id, NodeType.EXTRINSIC)
        self.optimizer.add_pose_prior(Trc_init, self.camera.id, NodeType.EXTRINSIC, extrinsic_prior_sigmas)

        self.Trc_traj: list[Geometry.SE3] = []
        last_timestep = None
        last_state = None
        last_HA_i = None
        last_HA_timestep = None
        for i, (timestep, state) in enumerate(self.states.items()):
            is_keyframe = timestep in self.kf_map
            extra_optims = 1
            # PERFORM POSE / VELOCITY SLAM ON ALL FRAMES
            Trc = self.optimizer.get_node_estimate(self.camera.id, NodeType.EXTRINSIC)
            if Trc is None: Trc = Trc_init
            self.Trc_traj.append(Trc)

            Twr = state.pose
            Twc = Twr.compose(Trc_init)
            ref_sigmas = state.sigmas
            self.optimizer.add_node(Twc, i, NodeType.CAMERA)

            if i % anchor_intervals == 0 and anchor_intervals > 0:
                self.optimizer.add_reference_anchor(self.camera.id, i, Twr, ref_sigmas)

            if i > 0:
                odom_r = Geometry.SE3.between(last_state.pose, state.pose)
                odom_c = Trc_init.inverse() * odom_r * Trc_init
                self.optimizer.add_between_factor(i-1, i, NodeType.CAMERA, odom_c, between_sigma)

            if i < n_priors or n_priors < 0:
                prior_value = Twr.compose(Trc_init)
                self.optimizer.add_pose_prior(prior_value, i, NodeType.CAMERA, prior_sigmas)

            if i % HA_interval == 0 and HA_interval > 0:
                if not last_HA_i:
                    last_HA_i = i
                    last_HA_timestep = timestep
                state_from = self.states[last_HA_timestep]
                self.optimizer.add_hand_eye_factor(last_HA_i, i, self.camera.id, state_from, state, NodeType.CAMERA, HA_sigmas)

            # Add reprojection of map points only for keyframes
            if is_keyframe and use_mps:
                kf_id = self.kf_map[timestep]
                keyframe = self.keyframes[kf_id]
                add_optim = False
                for map_point in self.map_points.values():
                    observations = map_point['observations']
                    if len(observations) < 3: continue # Skip map points with few observations
                    if kf_id not in observations.keys(): continue # Map point not in frame
                    mp_id = int(map_point['id'])
                    kp = keyframe['keypoints_und'][observations[kf_id]]
                    self.optimizer.add_smart_projection_factor(mp_id, kp, i, self.camera)
                    add_optim = True
                # if add_optim: extra_optims += 1
            
            
            try:
                self.optimizer.optimize(extra_optims) # Optimize after at least two timesteps have passed
            except Exception as e:
                print(e)
                break
            finally:
                print(i)

            last_timestep = timestep
            last_state = state

    def plot_gt(self, t, pos_axs: list[plt.Axes], ang_axs: list[plt.Axes]) -> None:
        ref_traj = np.empty([len(t), 6])
        cam_traj = np.empty([len(t), 6])
        for i, (timestep, state) in enumerate(self.states.items()):
            # print(kf_id, Time.TimeConversion.POSIX_to_STX(keyframe['timestep']))
            Twr = state.pose
            Twc = Twr.compose(self.Trc_gt)
            ref_traj[i, :] = Geometry.SE3.as_vector(Twr)
            cam_traj[i, :] = Geometry.SE3.as_vector(Twc)

        for i in range(len(pos_axs)):
            ang_axs[i].plot(t, ref_traj[:, i], label='Ref-gt')
            ang_axs[i].plot(t, cam_traj[:, i], label='Cam-gt')

            pos_axs[i].plot(t, ref_traj[:, i+3], label='Ref-gt')
            pos_axs[i].plot(t, cam_traj[:, i+3], label='Cam-gt')


    def plot_estim(self, t, pos_axs: list[plt.Axes], ang_axs: list[plt.Axes]) -> None:
        ref_traj = np.empty([len(t), 6])
        cam_traj = np.empty([len(t), 6])
        Trc_estim = self.optimizer.get_node_estimate(self.camera.id, NodeType.EXTRINSIC)
        for i, (timestep, state) in enumerate(self.states.items()):
            # ref_pose = self.optimizer.get_pose_node_estimate(kf_id_int, NodeType.REFERENCE)
            Twc = self.optimizer.get_node_estimate(i, NodeType.CAMERA)
            if not Twc:
                ref_traj[i, :] = np.nan
                cam_traj[i, :] = np.nan
                continue
            Twr = Twc.compose(Trc_estim.inverse())
            ref_traj[i, :] = Geometry.SE3.as_vector(Twr)
            cam_traj[i, :] = Geometry.SE3.as_vector(Twc)

        for i in range(len(pos_axs)):
            ang_axs[i].plot(t, ref_traj[:, i], label='Ref-estim')
            ang_axs[i].plot(t, cam_traj[:, i], label='Cam-estim')

            pos_axs[i].plot(t, ref_traj[:, i+3], label='Ref-estim')
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
        _, extr_axs = plt.subplots(2, 3)
        pos_axs, ang_axs = axs[0], axs[1]
        pos_axs_ext, ang_axs_ext = extr_axs[0], extr_axs[1]
        t = np.arange(len(self.states))
        self.plot_gt(t, pos_axs, ang_axs)
        self.plot_estim(t, pos_axs, ang_axs)
        self.plot_extrinsics(t, pos_axs_ext, ang_axs_ext)

        ang_labels = ['roll', 'pitch', 'yaw']
        pos_labels = ['North [m]', 'East [m]', 'Height [m]']
        for i in range(3):
            pos_axs[i].grid()
            pos_axs[i].legend()
            pos_axs[i].set_title(pos_labels[i])
            ang_axs[i].grid()
            ang_axs[i].legend()
            ang_axs[i].set_title(ang_labels[i])

            pos_axs_ext[i].grid()
            pos_axs_ext[i].legend()
            pos_axs_ext[i].set_title(pos_labels[i])
            ang_axs_ext[i].grid()
            ang_axs_ext[i].legend()
            ang_axs_ext[i].set_title(ang_labels[i])
            


        plt.show()
            
if __name__ == '__main__':

    app = Application()
    app.start()
    estim = app.optimizer.get_node_estimate(app.camera.id, NodeType.EXTRINSIC)
    gt = app.Trc_gt
    print(f'Estimated Trc: {Geometry.SE3.as_vector(estim)}')
    print(f'Ground truth Trc: {Geometry.SE3.as_vector(gt)}')
    graph, estimate = app.optimizer.get_visualization_variables()
    # Visualization.FactorGraphVisualization.draw_factor_graph('./output/', graph, estimate)

    app.show()
