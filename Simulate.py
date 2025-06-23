
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


def visual_ISAM2_plot(result, node_type, delay):
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

    gtsam_plot.plot_3d_points(fignum, result, 'rx')

    # Plot cameras
    i = 0
    while result.exists(node_type(i)):
        pose_i = result.atPose3(node_type(i))
        gtsam_plot.plot_pose3(fignum, pose_i, 10)
        i += 1

    # draw
    axes.set_xlim3d(-40, 40)
    axes.set_ylim3d(-40, 40)
    axes.set_zlim3d(-40, 40)
    plt.pause(delay)


def boat_movement(steps=100, delta_t=0.1) -> list[Geometry.SE3]:
    trajectory = []

    radius = 30.0  # meters
    angular_speed = 2 * np.pi / (steps * delta_t)  # one full circle
    roll_amp = 0.1      # radians
    pitch_amp = 0.05    # radians
    bobbing_freq = 0.5  # Hz (frequency of motion)
    z_amp = 0.3         # meters (wave height)
    sway_amp = 0.2      # meters (sideways motion)

    for i in range(steps):
        t = i * delta_t
        theta = np.pi - angular_speed * t  # start at (30, 0), CCW

        # Base circular position
        base_x = radius * np.cos(theta)
        base_y = radius * np.sin(theta)

        # Tangent unit vector (direction of motion)
        dx = -np.sin(theta)
        dy = np.cos(theta)

        # Normal vector (lateral direction pointing outward)
        nx = -dy
        ny = dx

        # Add lateral sway in normal direction
        sway = sway_amp * np.sin(2 * np.pi * bobbing_freq * t + np.pi / 3)
        x = base_x + sway * nx
        y = base_y + sway * ny

        # Z bobbing due to waves
        z = z_amp * np.sin(2 * np.pi * bobbing_freq * t)

        # Oscillatory roll and pitch
        roll = roll_amp * np.sin(2 * np.pi * bobbing_freq * t)
        pitch = pitch_amp * np.sin(2 * np.pi * bobbing_freq * t + np.pi / 4)

        # Yaw: facing tangent to path
        yaw = theta + np.pi / 2

        # Construct Pose
        R = Geometry.SO3.RzRyRx(roll, pitch, yaw)
        t = [x, y, z]

        trajectory.append(Geometry.SE3.from_vector(np.array([roll, pitch, yaw, x, y, z])))

    return trajectory


def boat_straight_movement(steps=100) -> list[Geometry.SE3]:
    trajectory = []

    start_x = -30.0  # match circular motion start
    start_y = -30.0
    start_z = 0
    end_x = 30.0
    total_distance = end_x - start_x
    delta_x = total_distance / (steps - 1)  # to include both start and end points

    for i in range(steps):
        x = start_x + i * delta_x
        y = start_y
        z = start_z

        roll = 180.0
        pitch = 0.0
        yaw = 0.0

        trajectory.append(Geometry.SE3.from_vector(np.array([roll, pitch, yaw, x, y, z]), radians=False))

    return trajectory



def generate_mps() -> list[Geometry.Point3]:
    mps = [
        np.array([10, 10, 10]),
        np.array([-10, 10, 10]),
        np.array([10, -10, 10]),
        np.array([-10, -10, 10]),
        np.array([10, 10, -10]),
        np.array([-10, 10, -10]),
        np.array([10, -10, -10]),
        np.array([-10, -10, -10]),
        np.array([5, 5, 5]),
        np.array([-5, 5, 5]),
        np.array([5, -5, 5]),
        np.array([-5, -5, 5]),
        np.array([5, 5, -5]),
        np.array([-5, 5, -5]),
        np.array([5, -5, -5]),
        np.array([-5, -5, -5]),
    ]
    return mps


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
        self.traj = boat_movement()
        self.traj = boat_straight_movement()
        self.mps = generate_mps()
        self.mps = [self.mps[0]]

        self.camera = Camera([50., 50., 50., 50.], [])

    def start(self):
        n_priors = 2
        plt.ion()

        prior_sigmas = np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1])

        init_noise = Geometry.SE3.from_vector(np.array([1, 2, -1, 0.1, -0.05, 0.2]), radians=False)
        Trc_noise = Geometry.SE3.from_vector(np.array([1, 0.5, -0.6, 0.1, 0.1, -0.1]), radians=False)
        landmark_noise = np.array([0.3, -0.2, 0.2])

        self.Trc = Geometry.SE3(Rotate.ROLL(-90).rotation(), [0, 0, 5])
        Trc_init = self.Trc.compose(Trc_noise)

        self.optimizer.add_node(Trc_init, self.camera.id, NodeType.EXTRINSIC)
        self.optimizer.add_prior(Trc_init, self.camera.id, NodeType.EXTRINSIC, prior_sigmas)

        for i, Twr in enumerate(self.traj):
            extra_optims = 0
            
            Twc = Twr.compose(self.Trc)
            Twc_noisy = Twc.compose(init_noise)
            self.optimizer.add_node(Twr, i, NodeType.REFERENCE)
            self.optimizer.add_pose_equality(Twr, i, NodeType.REFERENCE)
            # self.optimizer.add_node(Twc_noisy, i, NodeType.CAMERA)

            # if i < n_priors:
                # self.optimizer.add_prior(Twr, i, NodeType.REFERENCE, prior_sigmas)

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
            
            # visual_ISAM2_plot(self.optimizer.current_estimate, NodeType.REFERENCE, 0.1)
        
        plt.ioff()
        plt.show()

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

    # app.summarize_landmarks()
    app.check_ambig()
    
    graph, estimate = app.optimizer.get_visualization_variables()
    Visualization.FactorGraphVisualization.draw_factor_graph('./output/', graph, estimate, exclude_mps = False)

    app.show()
    
