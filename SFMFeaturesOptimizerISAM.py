
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
    
    def __init__(self, start_id=0, iterations=-1, visualize=False) -> None:
        self.plot_3d = None
        if visualize: self.plot_3d = LiveTrajectory3D(NodeType.REFERENCE, delay=0.01)
        relin_skip = 1
        relin_thres = 0.1
        # self.optimizer = Optimizer(relin_skip=relin_skip, relin_thres=relin_thres)
        self.optimizer = Optimizer()
        self.camera, self.Trc_gt = DataLoader.load_stx_camera()

        self.Trc_gt = self.Trc_gt.compose(Geometry.SE3.NED_to_RDF_map())

        noisy_vals = np.append([-1, 1, -0.3], [1.5, -2.5, 1.5])
        Trc_init = self.Trc_gt
        # Trc_init = Trc_init.compose(Geometry.SE3.from_vector(noisy_vals, radians=False))
        self.Trc_init = Trc_init

        self.load_gt()
        self.load_COLMAP()
        self.align_mps()
        self.filter_mps()
        
        length = len(self.colmap_obs)
        if start_id >= length:
            print('Start ID higher than number of images')
            exit()
        
        if start_id + iterations >= length or iterations <= 0:
            iterations = length - start_id

        self.start_id = start_id
        self.iterations = iterations


    def load_COLMAP(self):
        COLMAP_project_path = DataLoader.reconstrucion_folder()
        sparse_model_path = f'{COLMAP_project_path}/sparse/'
        self.reconstruction = Reconstruction(sparse_model_path)

        max_error = 1.5
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
        Trc = self.Trc_gt

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
        global_region = 400 # [m]
        local_dist = 100 # [m], max dist from camera

        # Remove based on distance
        new_mps_dict = {}
        for mp_id, point3d in self.colmap_mps.items():
            if np.linalg.norm(point3d) > global_region: continue
            new_mps_dict[mp_id] = point3d
        self.colmap_mps = new_mps_dict


        # Update per-image observation list
        new_obs_dict = {}
        for timestep, observations in self.colmap_obs.items():
            new_obs_list = []
            for (pixels, mp_id) in observations:
                if mp_id not in list(self.colmap_mps.keys()): continue

                new_obs_list.append((pixels, mp_id))
            new_obs_dict[timestep] = new_obs_list
        
        self.colmap_obs = new_obs_dict




    def start(self):
        extrinsic_prior_sigmas = np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1])*1e3

        self.optimizer.add_node(self.Trc_init, self.camera.id, NodeType.EXTRINSIC)
        self.optimizer.add_prior(self.Trc_init, self.camera.id, NodeType.EXTRINSIC, extrinsic_prior_sigmas)

        self.Trc_traj: list[Geometry.SE3] = []
        timesteps = self.colmap_obs.keys()

        pb = ProgressBar(self.iterations)
        print(f'Processing {self.iterations} images...')
        pb.print(0)

        for i, timestep in enumerate(sorted(timesteps)[self.start_id:self.start_id + self.iterations]):
            extra_optims = 0

            Trc = self.optimizer.get_node_estimate(self.camera.id, NodeType.EXTRINSIC)
            if Trc is None: Trc = self.Trc_init
            self.Trc_traj.append(Trc)
            
            state = self.states[timestep]
            Twr = state.pose

            self.optimizer.add_node(Twr, i, NodeType.REFERENCE)
            self.optimizer.add_pose_equality(Twr, i, NodeType.REFERENCE)

            for (pixels, mp_id) in self.colmap_obs[timestep]:
                # break
                point3d = self.colmap_mps[mp_id]
                if self.optimizer.get_node_estimate(mp_id, NodeType.LANDMARK) is None: self.optimizer.add_node(point3d, mp_id, NodeType.LANDMARK)
                self.optimizer.add_extrinsic_projection_factor(mp_id, pixels, i, self.camera) # Add observation to fg

            # if i > 0:
                # self.optimizer.optimize(extra_optims) # Optimize after at least two timesteps have passed

            if self.plot_3d is not None: self.plot_3d.update(self.optimizer.current_estimate)

            pb.print(i+1)

        print('Processing finished')
        if self.plot_3d is not None: self.plot_3d.finished()

    def plot_gt(self, t, pos_axs: list[plt.Axes], ang_axs: list[plt.Axes]) -> None:
        ref_traj = np.zeros([len(t), 6])
        timesteps = self.colmap_obs.keys()

        for i, timestep in enumerate(sorted(timesteps)[self.start_id:self.start_id + self.iterations]):
            if i == len(t): break
            Twr = self.states[timestep].pose

            ref_traj[i, :] = Geometry.SE3.as_vector(Twr)

        for i in range(len(pos_axs)):
            ang_axs[i].plot(t, ref_traj[:, i], label='Ref-gt')

            pos_axs[i].plot(t, ref_traj[:, i+3], label='Ref-gt')


    def plot_estim(self, t, pos_axs: list[plt.Axes], ang_axs: list[plt.Axes]) -> None:
        ref_traj = np.zeros([len(t), 6])

        for i in range(len(t)):
            Twr = self.optimizer.get_node_estimate(i, NodeType.REFERENCE)
            if Twr is None: ref_traj[i, :] = np.nan
            else: ref_traj[i, :] = Geometry.SE3.as_vector(Twr)

        for i in range(len(pos_axs)):
            ang_axs[i].plot(t, ref_traj[:, i], label='Ref-estim')

            pos_axs[i].plot(t, ref_traj[:, i+3], label='Ref-estim')


    def plot_extrinsics(self, t, pos_axs: list[plt.Axes], ang_axs: list[plt.Axes]) -> None:
        traj = np.zeros([len(t), 6])
        traj_gt = np.zeros([len(t), 6])

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
        t = np.arange(len(self.Trc_traj))
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

    app = Application(start_id=50, iterations=50, visualize=True)
    app.start()
    
    estim = app.optimizer.get_node_estimate(app.camera.id, NodeType.EXTRINSIC)
    # app.optimizer.factor_db.prepare_update()
    # result = app.optimizer.bundle_adjustment()
    # estim = result.atPose3(NodeType.EXTRINSIC(app.camera.id))
    gt = app.Trc_gt
    print(f'Estimated Trc: {Geometry.SE3.as_vector(estim)}')
    print(f'Ground truth Trc: {Geometry.SE3.as_vector(gt)}')

    graph, estimate = app.optimizer.get_visualization_variables()
    # Visualization.FactorGraphVisualization.draw_factor_graph('./output/', graph, estimate, exclude_mps = False)

    app.show()