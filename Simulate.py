
from src.backend import Optimizer, NodeType
from src.structs import Camera
from src.util import Geometry, DataLoader, Time
from src.visualization import Visualization

import numpy as np
from pycolmap import Reconstruction

from gtsam.examples import SFMdata
import gtsam.utils.plot as gtsam_plot
from gtsam.symbol_shorthand import L, X, C

import matplotlib.pyplot as plt


def visual_ISAM2_plot(result):
    """
    VisualISAMPlot plots current state of ISAM2 object
    Author: Ellon Paiva
    Based on MATLAB version by: Duy Nguyen Ta and Frank Dellaert
    """

    # Declare an id for the figure
    fignum = 0

    fig = plt.figure(fignum)
    if not fig.axes:
        axes = fig.add_subplot(projection='3d')
    else:
        axes = fig.axes[0]
    plt.cla()

    # Plot points
    # Can't use data because current frame might not see all points
    # marginals = Marginals(isam.getFactorsUnsafe(), isam.calculateEstimate())
    # gtsam.plot_3d_points(result, [], marginals)
    gtsam_plot.plot_3d_points(fignum, result, 'rx')

    # Plot cameras
    i = 0
    while result.exists(C(i)):
        pose_i = result.atPose3(C(i))
        gtsam_plot.plot_pose3(fignum, pose_i, 10)
        i += 1

    # draw
    axes.set_xlim3d(-40, 40)
    axes.set_ylim3d(-40, 40)
    axes.set_zlim3d(-40, 40)
    plt.pause(1)


class Rotate:
    
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
        self.optimizer = Optimizer()
        self.traj = self.generate_trajectory()
        # self.traj = SFMdata.createPoses(steps=30)
        # self.mps = self.generate_mps()
        self.mps = SFMdata.createPoints()

        self.camera = Camera([50., 50., 50., 50.], [])

    def generate_trajectory(self) -> list[Geometry.SE3]:
        start = Geometry.SE3(Rotate.YAW(180).rotation(), [30, 30, 0])
        mag = 30

        steps = [
            [Move.FORWARD] * mag,
            [Move.FORWARD] * mag,
            [Rotate.YAW(90)],
            [Move.FORWARD] * mag,
            [Move.FORWARD] * mag,
            [Rotate.YAW(90)],
            [Move.FORWARD] * mag,
            [Move.FORWARD] * mag,
            [Rotate.YAW(90)],
            [Move.FORWARD] * mag,
            [Move.FORWARD] * mag
        ]

        traj = [start]
        last_pose = start
        for iteration in steps:
            total_odom = Geometry.SE3()
            for step in iteration:
                total_odom = total_odom.compose(step)
            new_pose = last_pose.compose(total_odom)
            traj.append(new_pose)
            last_pose = new_pose

        return traj

    def generate_mps(self) -> list[Geometry.Point3]:
        mps = [
            np.array([10, 10, 10]),
            np.array([-10, 10, 10]),
            np.array([10, -10, 10]),
            np.array([-10, -10, 10]),
            np.array([10, 10, -10]),
            np.array([-10, 10, -10]),
            np.array([10, -10, -10]),
            np.array([-10, -10, -10]),
        ]
        return mps

    def start(self):
        n_priors = 2
        plt.ion()


        prior_sigmas = np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1])
        init_noise = Geometry.SE3.from_vector(np.array([1, 2, -1, 0.1, -0.05, 0.2]), radians=False)

        Trc = Geometry.SE3()
        for i, Twr in enumerate(self.traj):
            extra_optims = 1
            
            Twc = Twr.compose(Trc)
            Twc_noisy = Twc.compose(init_noise)
            self.optimizer.add_node(Twc_noisy, i, NodeType.CAMERA)

            if i < n_priors:
                self.optimizer.add_pose_prior(Twc_noisy, i, NodeType.CAMERA, prior_sigmas)

            for j, point in enumerate(self.mps):
                # if self.optimizer.get_node_estimate(mp_id, NodeType.LANDMARK) is None: self.optimizer.add_node(point3d, mp_id, NodeType.LANDMARK)
                # self.optimizer.add_projection_factor(mp_id, pixels, i, self.camera)
                point_cam = Twc.transformTo(point)
                pixels = self.camera.project(point_cam)
                Tc_c = Geometry.SE3()
                self.optimizer.add_smart_projection_factor(j, pixels, i, self.camera, Tc_c)

            try:
                if i > 0:
                    self.optimizer.optimize(extra_optims) # Optimize after at least two timesteps have passed
            except Exception as e:
                print(e)
                break
            finally:
                # if i == 5: break
                print(i)
            
            visual_ISAM2_plot(self.optimizer.current_estimate)
        
        plt.ioff()
        plt.show()

    def get_estim_traj(self) -> list[Geometry.SE3]:
        traj = []
        for i in range(len(self.traj)):
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
        t = np.arange(len(self.traj))
        gt_traj = self.traj
        estim_traj = self.get_estim_traj()
        error_traj = self.get_error_traj(gt_traj, estim_traj)
        self.plot_SE3(t, gt_traj, pos_axs, ang_axs, 'Cam-gt')
        self.plot_SE3(t, estim_traj, pos_axs, ang_axs, 'Cam-estim')
        self.plot_SE3(t, error_traj, pos_err_axs, ang_error_axs, 'Error')

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
    # Visualization.FactorGraphVisualization.draw_factor_graph('./output/', graph, estimate, exclude_mps = True)

    app.show()
    
