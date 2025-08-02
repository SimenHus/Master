
from src.backend import Optimizer, NodeType
from src.structs import Camera
from src.util import Geometry, DataLoader, Time, Geode
from src.util.ProgressBar import ProgressBar
from src.visualization import Visualization, LiveTrajectory3D


import json
import numpy as np
from pycolmap import Reconstruction


import matplotlib.pyplot as plt

class Application:
    
    def __init__(self, start_id=0, iterations=-1) -> None:

        self.optimizer = Optimizer()
        self.camera, self.Trc_gt = DataLoader.load_stx_camera()

        self.Trc_gt = self.Trc_gt.compose(Geometry.SE3.NED_to_RDF_map())

        noisy_vals = np.append([-10, 14, -23.], [3.2, -4.25, -3.15])
        # noisy_vals = np.append([-5, 4, -2.3], [0, 0, 0])
        Trc_init = self.Trc_gt
        Trc_init = Trc_init.compose(Geometry.SE3.from_vector(noisy_vals, radians=False))
        self.Trc_init = Trc_init

        self.load_gt()
        self.load_COLMAP()
        self.align_mps()
        
        length = len(self.colmap_obs)
        if start_id >= length:
            print('Start ID higher than number of images')
            exit()
        
        if start_id + iterations >= length or iterations <= 0:
            iterations = length - start_id

        timesteps = self.colmap_obs.keys()
        self.active_window = sorted(timesteps)[start_id:start_id + iterations]

        self.filter_mps()

    def load_COLMAP(self):
        COLMAP_project_path = DataLoader.reconstrucion_folder()
        sparse_model_path = f'{COLMAP_project_path}/sparse/'
        self.reconstruction = Reconstruction(sparse_model_path)

        max_error = 1.0
        min_obs = 30
        max_obs = -1

        self.colmap_pose = {} # Timestep: pose
        self.colmap_mps = {} # mp_id: point3d
        self.colmap_obs = {} # Timestep: list[(pixels, mp_id)]
        for img_id in self.reconstruction.images:
            img = self.reconstruction.image(img_id)
            timestep = Time.TimeConversion.generic_to_POSIX(img.name.strip('.jpg'))

            rot = Geometry.SO3(img.cam_from_world.rotation.matrix())
            trans = img.cam_from_world.translation
            pose = Geometry.SE3(rot, trans).inverse()

            observations = img.get_observation_points2D()
            obs_list = []
            for obs in observations:
                point = self.reconstruction.point3D(obs.point3D_id)
                point3d = point.xyz
                error = point.error
                n_obs = point.track.length()

                if error > max_error: continue
                if n_obs < min_obs: continue
                if max_obs > 0 and n_obs > max_obs: continue

                if obs.point3D_id not in self.colmap_mps:
                    self.colmap_mps[obs.point3D_id] = point3d

                obs_list.append((obs.xy, obs.point3D_id))

            self.colmap_obs[timestep] = obs_list
            self.colmap_pose[timestep] = pose


    def load_gt(self):
        stx_data = DataLoader.load_stx_data()
        filtered_data = stx_data

        sorted_data = sorted(filtered_data, key=lambda x: x.timestep)

        self.states: dict[int: Geometry.State] = {}

        for data in sorted_data:
            self.states[data.timestep] = data.state

    def align_mps(self):
        Trc = self.Trc_init

        NED_poses = []
        COLMAP_poses = []
        for timestep, colmap_pose in self.colmap_pose.items():

            ned_pose = self.states[timestep].pose.compose(Trc)
            NED_poses.append(ned_pose)
            COLMAP_poses.append(colmap_pose)

        c, T_align = Geometry.SE3.align_frames(COLMAP_poses, NED_poses)
        for mp_id, point3d in self.colmap_mps.items():
            self.colmap_mps[mp_id] = c * T_align.rotation().matrix() @ point3d + T_align.translation()

    def filter_mps(self):
        global_region = 100 # [m]
        x = 30
        obs_thresh = len(self.active_window) // x # Observed in at least x% of the sequence

        # Remove based on distance
        new_mps_dict = {}
        for mp_id, point3d in self.colmap_mps.items():
            if np.linalg.norm(point3d) > global_region: continue
            new_mps_dict[mp_id] = point3d
        self.colmap_mps = new_mps_dict

        # Register number of times each landmark is observed in the sequence
        observations = {}
        for timestep in self.active_window:
            for (pixels, mp_id) in self.colmap_obs[timestep]:
                if mp_id not in observations: observations[mp_id] = 0
                observations[mp_id] += 1

        # Update per-image observation list
        new_obs_dict = {}
        for timestep in self.active_window:
            new_obs_list = []
            for (pixels, mp_id) in self.colmap_obs[timestep]:
                if mp_id not in list(self.colmap_mps.keys()): continue
                if observations[mp_id] < obs_thresh: continue

                new_obs_list.append((pixels, mp_id))
            new_obs_dict[timestep] = new_obs_list
        
        self.colmap_obs = new_obs_dict


    def start(self):
        rot_sigmas = np.array([0.1]*3)*1e3
        pos_sigmas = np.array([0.1]*3)*1e3
        extrinsic_prior_sigmas = np.append(rot_sigmas, pos_sigmas)

        self.optimizer.add_node(self.Trc_init, self.camera.id, NodeType.EXTRINSIC)
        self.optimizer.add_prior(self.Trc_init, self.camera.id, NodeType.EXTRINSIC, extrinsic_prior_sigmas)

        pb = ProgressBar(len(self.active_window))
        print(f'Processing {len(self.active_window)} images...')
        pb.print(0)

        factors = 0
        for i, timestep in enumerate(self.active_window):
            
            state = self.states[timestep]
            Twr = state.pose

            self.optimizer.add_node(Twr, i, NodeType.REFERENCE)
            # self.optimizer.add_pose_equality(Twr, i, NodeType.REFERENCE)
            self.optimizer.add_prior(Twr, i, NodeType.REFERENCE, np.array([1e-2]*6))

            for (pixels, mp_id) in self.colmap_obs[timestep]:
                point3d = self.colmap_mps[mp_id]
                if self.optimizer.get_node_estimate(mp_id, NodeType.LANDMARK) is None: self.optimizer.add_node(point3d, mp_id, NodeType.LANDMARK)
                self.optimizer.add_extrinsic_projection_factor(mp_id, pixels, i, self.camera) # Add observation to fg
                factors += 1

            pb.print(i+1)

        print(f'Observations: {factors}')
        print('Performing bundle adjustment...')
        app.optimizer.factor_db.prepare_update()
        self.result = app.optimizer.bundle_adjustment()
        print('Processing finished')



    def plot_gt(self, t, pos_axs: list[plt.Axes], ang_axs: list[plt.Axes]) -> None:
        ref_traj = np.zeros([len(t), 6])

        for i, timestep in enumerate(self.active_window):
            if i == len(t): break
            Twr = self.states[timestep].pose

            ref_traj[i, :] = Geometry.SE3.as_vector(Twr, radians=False)

        for i in range(len(pos_axs)):
            ang_axs[i].plot(t, ref_traj[:, i], label='Ref-gt')

            pos_axs[i].plot(t, ref_traj[:, i+3], label='Ref-gt')


    def plot_estim(self, t, pos_axs: list[plt.Axes], ang_axs: list[plt.Axes]) -> None:
        ref_traj = np.zeros([len(t), 6])
        result = self.result

        for i in range(len(t)):
            Twr = result.atPose3(NodeType.REFERENCE(i))
            if Twr is None: ref_traj[i, :] = np.nan
            else: ref_traj[i, :] = Geometry.SE3.as_vector(Twr, radians=False)

        for i in range(len(pos_axs)):
            ang_axs[i].plot(t, ref_traj[:, i], label='Ref-estim')

            pos_axs[i].plot(t, ref_traj[:, i+3], label='Ref-estim')


    def show(self) -> None:
        fig, axs = plt.subplots(2, 3)
        pos_axs, ang_axs = axs[0], axs[1]
        t = np.arange(len(self.active_window))
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
    # asko_therese: 350- or 0-200
    start = 0
    iters = 100
    app = Application(start_id=start, iterations=iters)
    app.start()
    
    graph, result = app.optimizer.factor_db.master_graph, app.result
    Trc_key = NodeType.EXTRINSIC(app.camera.id)


    # linear_graph = graph.linearize(result)
    # H, eta = linear_graph.hessian()
    # print(f'Hessian rank: {np.linalg.matrix_rank(H)}')
    # print(f'Hessian shape: {H.shape}')


    estim = result.atPose3(Trc_key)
    gt = app.Trc_gt

    estim = Geometry.SE3.RDF_to_NED(estim)
    gt = Geometry.SE3.RDF_to_NED(gt)
    se3_error = Geometry.SE3.localCoordinates(estim, gt)
    se3_error[:3] *= 180 / np.pi

    print(f'Estimated Trc: {Geometry.SE3.as_vector(estim, radians=False)}')
    print(f'Ground truth Trc: {Geometry.SE3.as_vector(gt, radians=False)}')
    print(f'SE3 error: {se3_error} [deg, m]')

    marginals = Optimizer.marginals(graph, result)
    Trc_cov = marginals.marginalCovariance(Trc_key)
    
    pos_cov = Trc_cov[3:, 3:]
    rot_cov = Trc_cov[:3, :3]

    pos_std = np.diag(pos_cov)
    rot_std = np.diag(rot_cov) * 180 / np.pi

    max_rot_std = np.sqrt(np.max(np.linalg.eigvalsh(rot_cov))) * 180 / np.pi
    max_pos_std = np.sqrt(np.max(np.linalg.eigvalsh(pos_cov)))


    print(f'Rotation std devs / Worst case: {rot_std} / {max_rot_std} [deg]')
    print(f'Position std devs / Worst case: {pos_std} / {max_pos_std} [m]')

    # graph, estimate = app.optimizer.get_visualization_variables()
    # Visualization.FactorGraphVisualization.draw_factor_graph('./output/', graph, estimate, exclude_mps = False)

    app.show()