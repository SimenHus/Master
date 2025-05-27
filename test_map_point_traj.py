

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

        self.camera, self.Trc_gt = DataLoader.load_stx_camera('./datasets/osl')

    def load_map_points(self, path):
        with open(path, 'r') as f: self.map_points = json.load(f)

    def load_keyframes(self, path):
        with open(path, 'r') as f: self.keyframes = json.load(f)

    def load_gt(self, path):
        stx_data = DataLoader.load_stx_data(path)
        keyframe_timesteps = [kf['timestep'] for kf in self.keyframes.values()]
        filtered_data = [data for data in stx_data if data.timestep in keyframe_timesteps]

        sorted_data = sorted(filtered_data, key=lambda x: x.timestep)

        self.states: dict[int: Geometry.State] = {}
        ref_ecef = None
        ref_lla = None
        for data in sorted_data:
            state = data.state
            if ref_ecef is None:
                ref_ecef = state.position
                ref_lla = data.lla
            NED = Geode.Transformation.ECEF_to_NED(state.position, ref_ecef, ref_lla)
            # state.position = NED
            self.states[data.timestep] = data.state


    def start(self):
        iter_before_optim = 1
        iter_before_mpts = 0
        n_priors = 2

        kinetic_sigmas = np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3]) * 1e1
        prior_sigmas = np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3]) * 1e-1
        between_sigmas = np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3]) * 1e-4

        noisy_vals = np.append(np.array([-1, 2, -4])*np.pi/180, [0.2, -0.2, 0.1])
        Trc_init_noise = Geometry.SE3.Expmap(noisy_vals)
        Trc_init = self.Trc_gt.compose(Trc_init_noise)
        # Trc_init = Geometry.SE3()

        self.optimizer.add_pose_node(Trc_init, self.camera.id, NodeType.EXTRINSIC)
        self.optimizer.add_pose_prior(Trc_init, self.camera.id, NodeType.EXTRINSIC, [1e1]*6)

        last_kf_id = 0
        last_timestep = None
        last_state = None
        for i, (kf_id, keyframe) in enumerate(self.keyframes.items()):
            kf_id_int = int(keyframe['id'])
            current_timestep = keyframe['timestep']

            Trc = self.optimizer.get_pose_node_estimate(self.camera.id, NodeType.EXTRINSIC)
            if Trc is None: Trc = Trc_init

            ref_state = self.states[current_timestep] # Get ref state gt from file
            ref_pose = ref_state.pose
            ref_sigmas = ref_state.sigmas
            cam_pose = ref_pose.compose(Trc) # Initial guess of cam pose

            # self.optimizer.add_pose_node(ref_pose, kf_id_int, NodeType.REFERENCE) # Add node for reference poses
            self.optimizer.add_pose_node(cam_pose, kf_id_int, NodeType.CAMERA) # Add node for camera poses
            # self.optimizer.add_pose_equality(ref_pose, kf_id_int, NodeType.REFERENCE)
            self.optimizer.add_reference_anchor(self.camera.id, kf_id_int, ref_pose, ref_sigmas)
            # self.optimizer.add_camera_between_factor(kf_id_int, self.camera.id, kf_id_int, between_sigmas)

            if i < n_priors:
                # self.optimizer.add_pose_prior(ref_pose, kf_id_int, NodeType.REFERENCE, prior_sigmas)
                self.optimizer.add_pose_prior(cam_pose, kf_id_int, NodeType.CAMERA, prior_sigmas)

            if i > 0:
                dt_posix = current_timestep - last_timestep
                dt = Time.TimeConversion.dt_POSIX_to_SEC(dt_posix)
                sigmas = kinetic_sigmas * dt
                self.optimizer.add_kinematic_factor(last_kf_id, kf_id_int, self.camera.id, NodeType.CAMERA, last_state, dt, sigmas)

            for map_point in self.map_points.values():
                observations = map_point['observations']
                if i < iter_before_mpts: break
                if len(observations) < 3: continue # Skip map points with few observations
                if kf_id not in observations.keys(): continue # Map point not in frame
                mp_id = int(map_point['id'])
                kp = keyframe['keypoints_und'][observations[kf_id]]
                # self.optimizer.update_projection_factor(mp_id, kp, kf_id_int, self.camera)
            
            
            try:
                if i >= iter_before_optim:
                    self.optimizer.optimize() # Optimize after at least two timesteps have passed
            except Exception as e:
                break
            finally:
                print(i)

            last_kf_id = kf_id_int
            last_timestep = keyframe['timestep']
            last_state = ref_state

    def plot_gt(self, t, pos_axs: list[plt.Axes], ang_axs: list[plt.Axes]) -> None:
        ref_traj = np.empty([len(self.keyframes), 6])
        cam_traj = np.empty([len(self.keyframes), 6])
        for i, (kf_id, keyframe) in enumerate(self.keyframes.items()):
            # print(kf_id, Time.TimeConversion.POSIX_to_STX(keyframe['timestep']))
            ref_state = self.states[keyframe['timestep']]
            ref_pose = ref_state.pose
            cam_pose = ref_pose.compose(self.Trc_gt)
            # vec = Geometry.SE3.as_vector(ref_pose)
            # if not np.allclose(vec[:3], ref_state.attitude*180/np.pi, rtol=1e-5):
            #     print(f'GT: {ref_state.attitude*180/np.pi}, Trans: {vec[:3]}')
            ref_traj[i, :] = Geometry.SE3.as_vector(ref_pose)
            cam_traj[i, :] = Geometry.SE3.as_vector(cam_pose)

        for i in range(len(pos_axs)):
            ang_axs[i].plot(t, ref_traj[:, i], label='Ref-gt')
            ang_axs[i].plot(t, cam_traj[:, i], label='Cam-gt')

            pos_axs[i].plot(t, ref_traj[:, i+3], label='Ref-gt')
            pos_axs[i].plot(t, cam_traj[:, i+3], label='Cam-gt')


    def plot_estim(self, t, pos_axs: list[plt.Axes], ang_axs: list[plt.Axes]) -> None:
        ref_traj = np.empty([len(self.keyframes), 6])
        cam_traj = np.empty([len(self.keyframes), 6])
        Trc_estim = self.optimizer.get_pose_node_estimate(self.camera.id, NodeType.EXTRINSIC)
        for i, (kf_id, keyframe) in enumerate(self.keyframes.items()):
            kf_id_int = int(kf_id)
            ref_pose = self.optimizer.get_pose_node_estimate(kf_id_int, NodeType.REFERENCE)
            cam_pose = self.optimizer.get_pose_node_estimate(kf_id_int, NodeType.CAMERA)
            if not cam_pose:
                ref_traj[i, :] = np.nan
                cam_traj[i, :] = np.nan
                continue
            ref_pose = cam_pose.compose(Trc_estim.inverse())
            ref_traj[i, :] = Geometry.SE3.as_vector(ref_pose)
            cam_traj[i, :] = Geometry.SE3.as_vector(cam_pose)

        for i in range(len(pos_axs)):
            ang_axs[i].plot(t, ref_traj[:, i], label='Ref-estim')
            ang_axs[i].plot(t, cam_traj[:, i], label='Cam-estim')

            pos_axs[i].plot(t, ref_traj[:, i+3], label='Ref-estim')
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
    estim = app.optimizer.get_pose_node_estimate(app.camera.id, NodeType.EXTRINSIC)
    gt = app.Trc_gt
    print(f'Estimated Trc: {Geometry.SE3.as_vector(estim)}')
    print(f'Ground truth Trc: {Geometry.SE3.as_vector(gt)}')
    graph, estimate = app.optimizer.get_visualization_variables()
    Visualization.FactorGraphVisualization.draw_factor_graph('./output/', graph, estimate)
    app.show()
    
