
from src.backend import Optimizer, NodeType
from src.structs import Camera
from src.util import Geometry, DataLoader, Time, Geode
from src.visualization import Visualization


import json
import numpy as np
from pycolmap import Reconstruction


import matplotlib.pyplot as plt

class Application:
    
    def __init__(self) -> None:
        self.optimizer = Optimizer()
        self.load_COLMAP()
        self.load_gt('./datasets/osl/data')

        self.camera, self.Trc_gt = DataLoader.load_stx_camera()

    def load_COLMAP(self):
        COLMAP_project_path = DataLoader.COLMAP_project_path()
        sparse_model_path = f'{COLMAP_project_path}/sparse/'
        self.reconstruction = Reconstruction(sparse_model_path)

        self.colmap_pose = {} # Timestep: pose
        self.colmap_mps = {} # Timestep: list[(pixels, mp_id, point3d)]
        for img_id in self.reconstruction.images:
            img = self.reconstruction.image(img_id)
            timestep = Time.TimeConversion.generic_to_POSIX(img.name.strip('.jpg'))

            rot = Geometry.SO3(img.cam_from_world.rotation.matrix())
            trans = img.cam_from_world.translation
            pose = Geometry.SE3(rot, trans)

            observations = img.get_observation_points2D()
            obs_list = []
            for obs in observations:
                point3d = self.reconstruction.point3D(obs.point3D_id)
                obs_list.append((obs.xy, obs.point3D_id, point3d.xyz))

            self.colmap_mps[timestep] = obs_list
            self.colmap_pose[timestep] = pose


    def load_gt(self, path):
        stx_data = DataLoader.load_stx_data()
        filtered_data = stx_data

        sorted_data = sorted(filtered_data, key=lambda x: x.timestep)

        self.states: dict[int: Geometry.State] = {}

        for data in sorted_data:
            self.states[data.timestep] = data.state


    def start(self):
        n_priors = 2
        anchor_intervals = -1
        HA_interval = -1
        between_interval = -1

        prior_sigmas = np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3]) * 1e-2
        extrinsic_prior_sigmas = np.append([1e-1]*3, [1e-3]*3)
        HA_sigmas = np.array([0.1, 0.1, 0.1, 0.03, 0.03, 0.03]) * 1e-1
        between_extrinsic_sigmas = np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3]) * 1e0

        noisy_vals = np.append([-1, 2, -4], [0.05, -0.05, 0.05]) * 1
        Trc_init = Geometry.SE3.as_vector(self.Trc_gt) + noisy_vals
        Trc_init = Geometry.SE3.from_vector(Trc_init, radians=False)

        self.optimizer.add_node(Trc_init, self.camera.id, NodeType.EXTRINSIC)
        self.optimizer.add_pose_prior(Trc_init, self.camera.id, NodeType.EXTRINSIC, extrinsic_prior_sigmas)

        self.Trc_traj: list[Geometry.SE3] = []
        timesteps = self.colmap_mps.keys()
        last_HA_i = None
        last_HA_timestep = None
        for i, timestep in enumerate(sorted(timesteps)):
            extra_optims = 1
            Trc = self.optimizer.get_node_estimate(self.camera.id, NodeType.EXTRINSIC)
            if Trc is None: Trc = Trc_init
            self.Trc_traj.append(Trc)
            
            # Twc = self.colmap_pose[timestep].inverse().compose(Geometry.SE3.NED_to_RDF_map().inverse())
            # Twc = self.colmap_pose[timestep].inverse()
            state = self.states[timestep]
            Twr = state.pose
            Twc = Twr.compose(Trc_init)
            # Twc = Twc.compose(Geometry.SE3.NED_to_RDF_map())
            self.optimizer.add_node(Twr, i, NodeType.REFERENCE)
            self.optimizer.add_node(Twc, i, NodeType.CAMERA)
            self.optimizer.add_pose_equality(Twr, i, NodeType.REFERENCE)

            if i % between_interval == 0 and between_interval > 0:
                self.optimizer.add_camera_between_factor(i, self.camera.id, i, between_extrinsic_sigmas)

            if i < n_priors:
                self.optimizer.add_pose_prior(Twc, i, NodeType.CAMERA, prior_sigmas)
            
            if i % anchor_intervals == 0 and anchor_intervals > 0:
                ref_sigmas = state.sigmas * 1e-3
                self.optimizer.add_reference_anchor(self.camera.id, i, Twr, ref_sigmas)

            if i % HA_interval == 0 and HA_interval > 0:
                if last_HA_i is not None:
                    state_from = self.states[last_HA_timestep]
                    self.optimizer.add_hand_eye_factor(last_HA_i, i, self.camera.id, state_from, state, NodeType.CAMERA, HA_sigmas)
                last_HA_i = i
                last_HA_timestep = timestep

            for (pixels, mp_id, point3d) in self.colmap_mps[timestep]:
                # if self.optimizer.get_node_estimate(mp_id, NodeType.LANDMARK) is None: self.optimizer.add_node(point3d, mp_id, NodeType.LANDMARK)
                # self.optimizer.add_projection_factor(mp_id, pixels, i, self.camera)
                Tc_c = None
                # Tc_c = Geometry.SE3()
                self.optimizer.add_smart_projection_factor(mp_id, pixels, i, self.camera, Tc_c)


            try:
                self.optimizer.optimize(extra_optims) # Optimize after at least two timesteps have passed
            except Exception as e:
                print(e)
                break
            finally:
                # if i == 1: break
                print(i)

    def plot_gt(self, t, pos_axs: list[plt.Axes], ang_axs: list[plt.Axes]) -> None:
        ref_traj = np.zeros([len(t), 6])
        cam_traj = np.zeros([len(t), 6])
        timesteps = self.colmap_mps.keys()

        Twc_init = self.optimizer.get_node_estimate(0, NodeType.CAMERA)

        Twr_prev = None
        Twc_prev = None
        Trc = self.Trc_gt
        # Trc = self.Trc_gt.compose(Geometry.SE3.NED_to_RDF_map())
        for i, timestep in enumerate(sorted(timesteps)):
            Twr = self.states[timestep].pose
            # Twc = Twr.compose(self.Trc_gt)
            if Twr_prev is None:
                Twc = Twc_init
            else:
                Tr1r2 = Geometry.SE3.between(Twr_prev, Twr)
                Tc1c2 = Trc.inverse() * Tr1r2 * Trc
                Twc = Twc_prev.compose(Tc1c2)
                
            ref_traj[i, :] = Geometry.SE3.as_vector(Twr)
            cam_traj[i, :] = Geometry.SE3.as_vector(Twc)

            Twr_prev = Twr
            Twc_prev = Twc

        for i in range(len(pos_axs)):
            ang_axs[i].plot(t, ref_traj[:, i], label='Ref-gt')
            ang_axs[i].plot(t, cam_traj[:, i], label='Cam-gt')

            pos_axs[i].plot(t, ref_traj[:, i+3], label='Ref-gt')
            pos_axs[i].plot(t, cam_traj[:, i+3], label='Cam-gt')


    def plot_estim(self, t, pos_axs: list[plt.Axes], ang_axs: list[plt.Axes]) -> None:
        ref_traj = np.zeros([len(t), 6])
        cam_traj = np.zeros([len(t), 6])
        timesteps = self.colmap_mps.keys()
        Trc = self.optimizer.get_node_estimate(self.camera.id, NodeType.EXTRINSIC)
        for i, timestep in enumerate(sorted(timesteps)):
            Twr = self.optimizer.get_node_estimate(i, NodeType.REFERENCE)
            Twc = self.optimizer.get_node_estimate(i, NodeType.CAMERA)
            if Twr is None: ref_traj[i, :] = np.nan
            else: ref_traj[i, :] = Geometry.SE3.as_vector(Twr)

            if Twc is None: cam_traj[i, :] = np.nan
            else: cam_traj[i, :] = Geometry.SE3.as_vector(Twc)

        for i in range(len(pos_axs)):
            ang_axs[i].plot(t, ref_traj[:, i], label='Ref-estim')
            ang_axs[i].plot(t, cam_traj[:, i], label='Cam-estim')

            pos_axs[i].plot(t, ref_traj[:, i+3], label='Ref-estim')
            pos_axs[i].plot(t, cam_traj[:, i+3], label='Cam-estim')


    def plot_extrinsics(self, t, pos_axs: list[plt.Axes], ang_axs: list[plt.Axes]) -> None:
        traj = np.zeros([len(t), 6])
        traj_gt = np.zeros([len(t), 6])

        gt = Geometry.SE3.as_vector(self.Trc_gt)
        for i, _ in enumerate(self.Trc_traj):
            Twr = self.optimizer.get_node_estimate(i, NodeType.REFERENCE)
            Twc = self.optimizer.get_node_estimate(i, NodeType.CAMERA)
            Trc = Geometry.SE3.between(Twr, Twc)
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
        t = np.arange(len(self.colmap_mps))
        self.plot_gt(t, pos_axs, ang_axs)
        self.plot_estim(t, pos_axs, ang_axs)
        self.plot_extrinsics(t, pos_axs_ext, ang_axs_ext)

        ang_labels = ['roll', 'pitch', 'yaw']
        pos_labels = ['x', 'y', 'z']
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
    Visualization.FactorGraphVisualization.draw_factor_graph('./output/', graph, estimate, exclude_mps = True)

    app.show()
    
