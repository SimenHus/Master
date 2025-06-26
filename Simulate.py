
from src.backend import Optimizer, NodeType
from src.structs import Camera
from src.util import Geometry, DataLoader, Time
from src.visualization import Visualization, LiveTrajectory3D
from src.simulation import TrajectoryGenerator, SeaStates, LandmarkGenerator

import numpy as np

import matplotlib.pyplot as plt



class Rotate:

    @classmethod
    def ROLL(clc, degs) -> Geometry.SE3:
        return Geometry.SE3(Geometry.SO3.Rx(degs*np.pi/180),  [0, 0, 0])
    
    @classmethod
    def PITCH(clc, degs) -> Geometry.SE3:
        return Geometry.SE3(Geometry.SO3.Ry(degs*np.pi/180), [0, 0, 0])
    
    @classmethod
    def YAW(clc, degs) -> Geometry.SE3:
        return Geometry.SE3(Geometry.SO3.Rz(degs*np.pi/180), [0, 0, 0])

class Move:
    FORWARD = Geometry.SE3(Geometry.SO3(), [1, 0, 0])
    RIGHT = Geometry.SE3(Geometry.SO3(), [0, 1, 0])
    BACKWARD = Geometry.SE3(Geometry.SO3(), [-1, 0, 0])
    LEFT = Geometry.SE3(Geometry.SO3(), [0, -1, 0])



class Application:
    
    def __init__(self) -> None:
        self.optimizer = Optimizer(relin_skip=1, relin_thres=0.1)
        traj_settings = SeaStates()
        self.traj = TrajectoryGenerator.circular(w=1., settings=traj_settings)
        self.traj = TrajectoryGenerator.circular(w=0., settings=traj_settings)
        self.mps = LandmarkGenerator.grid_mps()

        self.camera = Camera([50., 50., 50., 50.], [])

    def start(self):

        plot_3D = LiveTrajectory3D(NodeType.REFERENCE, delay=1.0)
        prior_sigmas = np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1])*1e3

        init_noise = Geometry.SE3.from_vector(np.array([1, 2, -1, 0.1, -0.05, 0.2]), radians=False)
        Trc_noise = Geometry.SE3.from_vector(np.array([5, 2.5, -8.6, 2.1, 1.1, -3.1]), radians=False)
        landmark_noise = np.array([0.3, -0.2, 0.2])

        self.Trc = Geometry.SE3(Rotate.ROLL(-90).rotation(), [0, 0, 5])
        Trc_init = self.Trc.compose(Trc_noise)
        print(Trc_init)

        self.optimizer.add_node(Trc_init, self.camera.id, NodeType.EXTRINSIC)
        self.optimizer.add_prior(Trc_init, self.camera.id, NodeType.EXTRINSIC, prior_sigmas)

        for i, Twr in enumerate(self.traj):
            extra_optims = 0
            
            Twc = Twr.compose(self.Trc)
            self.optimizer.add_node(Twr, i, NodeType.REFERENCE)
            self.optimizer.add_pose_equality(Twr, i, NodeType.REFERENCE)


            for j, point in enumerate(self.mps):
                point_cam = Twc.transformTo(point)
                pixels = self.camera.project(point_cam)
            
                if self.optimizer.get_node_estimate(j, NodeType.LANDMARK) is None: self.optimizer.add_node(point + landmark_noise, j, NodeType.LANDMARK)
                self.optimizer.add_extrinsic_projection_factor(j, pixels, i, self.camera)

            try:
                if i > 0:
                    self.optimizer.optimize(extra_optims) # Optimize after at least two timesteps have passed
            except Exception as e:
                print(e)
                break
            finally:
                # if i == 5: break
                print(i)
            
            plot_3D.update(self.optimizer.current_estimate)
        
        plot_3D.finished()
        

    def get_estim_traj(self, node_type) -> list[Geometry.SE3]:
        traj = []
        for i in range(len(self.traj)):
            Twc = self.optimizer.get_node_estimate(i, node_type)
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
        t = np.arange(len(self.traj))

        Trc = self.optimizer.get_node_estimate(self.camera.id, NodeType.EXTRINSIC)

        gt_traj_ref = self.traj
        gt_traj_cam = [pose.compose(self.Trc) for pose in gt_traj_ref]

        ref_estim_traj = self.get_estim_traj(NodeType.REFERENCE)
        cam_estim_traj = [Twr.compose(Trc) for Twr in ref_estim_traj if Twr is not None]

        error_traj_cam = self.get_error_traj(gt_traj_cam, cam_estim_traj)
        error_traj_ref = self.get_error_traj(gt_traj_ref, ref_estim_traj)

        self.plot_SE3(t, gt_traj_cam, pos_axs, ang_axs, 'Cam-gt')
        self.plot_SE3(t, cam_estim_traj, pos_axs, ang_axs, 'Cam-estim')

        self.plot_SE3(t, gt_traj_ref, pos_axs, ang_axs, 'Ref-gt')
        self.plot_SE3(t, ref_estim_traj, pos_axs, ang_axs, 'Ref-estim')

        self.plot_SE3(t, error_traj_cam, pos_err_axs, ang_error_axs, 'Cam-error')
        self.plot_SE3(t, error_traj_ref, pos_err_axs, ang_error_axs, 'Ref-error')

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


    def summarize_landmarks(self) -> None:
        landmarks = self.mps
        for i, gt in enumerate(landmarks):
            landmark_estim = self.optimizer.get_node_estimate(i, NodeType.LANDMARK)
            
            print(f'Landmark {i+1}:')
            print(f'GT = {gt}')
            print(f'Estim = {landmark_estim}')
            print(f'Error = {landmark_estim - gt}')
            print('\n')


    def check_ambig(self) -> None:
        poses = self.traj
        landmarks = self.mps
        Trc = self.Trc

        pose_estims = [self.optimizer.get_node_estimate(i, NodeType.REFERENCE) for i in range(len(poses))]
        landmark_estims = [self.optimizer.get_node_estimate(i, NodeType.LANDMARK) for i in range(len(landmarks))]
        Trc_estim = self.optimizer.get_node_estimate(self.camera.id, NodeType.EXTRINSIC)


        for i, landmark in enumerate(landmark_estims):
            pose = pose_estims[0]
            Twc_estim = pose.compose(Trc_estim)
            landmark_c_estim = Twc_estim.transformTo(landmark)
            landmark_c_gt = poses[0].compose(Trc).transformTo(landmarks[i])
            print(f'Landmark {i+1}:')
            print(f'GT = {landmarks[i]}')
            print(f'Estim = {landmark}')
            print(f'Landmark_c = {landmark_c_estim}')
            print(f'Landmark_c gt = {landmark_c_gt}')
            print(f'Landmark_c error = {landmark_c_estim - landmark_c_gt}')
            print('\n')
        
            
if __name__ == '__main__':

    app = Application()
    app.start()

    estim = app.optimizer.get_node_estimate(app.camera.id, NodeType.EXTRINSIC)
    gt = app.Trc
    error = Geometry.SE3.as_vector(estim) - Geometry.SE3.as_vector(gt)
    print(f'Estimated Trc: {Geometry.SE3.as_vector(estim)}')
    print(f'Ground truth Trc: {Geometry.SE3.as_vector(gt)}')
    print(f'Trc error: {error}')

    print(f'Estim: {estim}')
    print(f'GT: {gt}')

    # app.summarize_landmarks()
    # app.check_ambig()
    
    graph, estimate = app.optimizer.get_visualization_variables()
    Visualization.FactorGraphVisualization.draw_factor_graph('./output/', graph, estimate, exclude_mps = False)

    app.show()
    
