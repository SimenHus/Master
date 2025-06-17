
from src.backend import Optimizer, NodeType
from src.structs import Camera
from src.util import Geometry, DataLoader, Time
from src.visualization import Visualization

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

        prior_sigmas = np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3]) * 1e-2

        timesteps = self.colmap_mps.keys()
        for i, timestep in enumerate(sorted(timesteps)):
            extra_optims = 1
            
            # Twc = self.colmap_pose[timestep].inverse().compose(Geometry.SE3.NED_to_RDF_map().inverse())
            Twc = self.colmap_pose[timestep].inverse()

            self.optimizer.add_node(Twc, i, NodeType.CAMERA)

            if i < n_priors:
                self.optimizer.add_pose_prior(Twc, i, NodeType.CAMERA, prior_sigmas)

            for (pixels, mp_id, point3d) in self.colmap_mps[timestep]:
                # if self.optimizer.get_node_estimate(mp_id, NodeType.LANDMARK) is None: self.optimizer.add_node(point3d, mp_id, NodeType.LANDMARK)
                # self.optimizer.add_projection_factor(mp_id, pixels, i, self.camera)
                Tc_c = None
                Tc_c = Geometry.SE3()
                self.optimizer.add_smart_projection_factor(mp_id, pixels, i, self.camera, Tc_c)

            try:
                if i > 0:
                    self.optimizer.optimize(extra_optims) # Optimize after at least two timesteps have passed
            except Exception as e:
                print(e)
                break
            finally:
                # if i == 5: break
                print(i)

    def get_gt_traj(self) -> list[Geometry.SE3]:

        traj = []
        timesteps = self.colmap_mps.keys()

        Twr_prev = None
        Twc_prev = None
        Trc = self.Trc_gt
        Trc = Trc.compose(Geometry.SE3.NED_to_RDF_map())
        for i, timestep in enumerate(sorted(timesteps)):
            Twr = self.states[timestep].pose

            if Twr_prev is None:
                Twc = self.colmap_pose[timestep].inverse()
            else:
                Tr1r2 = Geometry.SE3.between(Twr_prev, Twr)
                Tc1c2 = Trc.inverse() * Tr1r2 * Trc
                Twc = Twc_prev.compose(Tc1c2)
                
            traj.append(Twc)

            Twr_prev = Twr
            Twc_prev = Twc
        return traj

    def get_estim_traj(self) -> list[Geometry.SE3]:
        traj = []
        timesteps = self.colmap_mps.keys()
        for i, timestep in enumerate(sorted(timesteps)):
            Twc = self.optimizer.get_node_estimate(i, NodeType.CAMERA)
            traj.append(Twc)
        return traj
    
    def get_error_traj(self, gt_traj, estim_traj) -> list[Geometry.SE3]:
        traj = []
        for gt, estim in zip(gt_traj, estim_traj):

            if gt is None or estim is None:
                traj.append(None)
                continue

            error = Geometry.SE3.between(gt, estim)
            traj.append(error)
        return traj


    def plot_SE3(self, t, poses: list[Geometry.SE3], pos_axs: list[plt.Axes], ang_axs: list[plt.Axes], label: str) -> None:
        traj = np.zeros([len(t), 6])
        for i, pose in enumerate(poses):
            if pose is None:
                traj[i, :] = np.nan
            else:
                traj[i, :] = Geometry.SE3.as_vector(pose)

        for i in range(len(pos_axs)):
            ang_axs[i].plot(t, traj[:, i], label=label)
            pos_axs[i].plot(t, traj[:, i+3], label=label)



    def show(self) -> None:
        fig, axs = plt.subplots(4, 3)
        pos_axs, ang_axs, pos_err_axs, ang_error_axs = axs
        t = np.arange(len(self.colmap_mps))
        gt_traj = self.get_gt_traj()
        estim_traj = self.get_estim_traj()
        error_traj = self.get_error_traj(gt_traj, estim_traj)
        self.plot_SE3(t, gt_traj, pos_axs, ang_axs, 'Cam-gt')
        self.plot_SE3(t, estim_traj, pos_axs, ang_axs, 'Cam-estim')
        self.plot_SE3(t, estim_traj, pos_err_axs, ang_error_axs, 'Error')

        labels = {
            0: ['x', 'y', 'z'],
            1: ['roll', 'pitch', 'yaw'],
            2: ['x', 'y', 'z'],
            3: ['roll', 'pitch', 'yaw']
        }

        for i, ax_row in enumerate(axs):
            for j, ax in enumerate(ax_row):
                ax.grid()
                ax.legend()
                ax.set_title(labels[i][j])

        plt.show()
            
if __name__ == '__main__':

    app = Application()
    app.start()
    
    graph, estimate = app.optimizer.get_visualization_variables()
    Visualization.FactorGraphVisualization.draw_factor_graph('./output/', graph, estimate, exclude_mps = True)

    data = [app.optimizer.get_node_estimate(i, NodeType.CAMERA) for i in range(len(app.colmap_mps))]
    data = [Geometry.SE3.as_vector(p) for p in data if p is not None]
    filename = f'{DataLoader.output_path()}/trajectory.csv'
    np.savetxt(filename, data, delimiter=',')

    app.show()
    
