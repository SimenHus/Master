import gtsam # https://gtbook.github.io/gtsam-examples/intro.html
from gtsam import NonlinearFactorGraph, Values, Symbol
from gtsam import Point3, Point2, Pose3, Pose2, Rot3, Rot2, Symbol

from graphviz import Source, Graph

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d

from src.util import Geometry


class FactorGraphVisualization:

    @staticmethod
    def format_to_graphviz(graph: NonlinearFactorGraph, values: Values, exclude_mps: bool) -> Graph:
        dot = Graph(engine='sfdp')

        for key in values.keys():
            sym = Symbol(key)
            name, index = chr(sym.chr()), str(sym.index())
            dot.node(name + index, shape='ellipse')

        for i in range(graph.size()):
            factor = graph.at(i)
            if not factor: continue

            fac_type = type(factor).__name__
            if exclude_mps and 'projection' in fac_type.lower(): continue

            factor_name = f'Factor {i}'
            dot.node(factor_name, fac_type, shape='box', style='filled')
            for key in factor.keys():
                sym = Symbol(key)
                name, index = chr(sym.chr()), str(sym.index())
                dot.edge(factor_name, name + index)
        return dot


    @classmethod
    def draw_factor_graph(clc, output_folder: str, graph: NonlinearFactorGraph, values: Values, filename: str = 'factor_graph', exclude_mps = False) -> None:
        s = clc.format_to_graphviz(graph, values, exclude_mps)
        s.render(filename, format='png', cleanup=True, directory=output_folder)


class ReportPlot:
    max_ylim_pos = 8
    max_ylim_rot = 5
    def __init__(self, t, Trc, Trc_results, covs):
        self.Trc = Trc
        self.Trc_arr = Geometry.SE3.as_vector(Trc, radians=False)
        self.Trc_results = Trc_results
        self.Trc_results_arr = np.array([Geometry.SE3.as_vector(pose, radians=False) for pose in Trc_results])
        self.covs = covs
        self.t = t

    def se3_error(self):
        error_list = []
        for Trc_estim in self.Trc_results:
            error_local = Geometry.SE3.localCoordinates(self.Trc, Trc_estim)
            error_world = Geometry.SE3.transform_twist(self.Trc, error_local)
            error_list.append(error_world)
        return np.array(error_list)
    
    def vector_estim(self):
        return np.array([Geometry.SE3.as_vector(estim, radians=False) for estim in self.Trc_results])
    

    def plot_conf2(self, t, conf_pos, conf_rot, xlabel) -> None:
        covs = self.covs
        Trc_estims = self.Trc_results
        pos_arr = []
        rot_arr = []
        for (cov, Trc_estim) in zip(covs, Trc_estims):

            if cov is not None: cov = Geometry.SE3.transform_cov(Trc_estim, cov)

            pos = np.sqrt(np.diag(cov[3:, 3:])) if cov is not None else np.array([np.inf]*3)
            rot = np.sqrt(np.diag(cov[:3, :3])) * 180 / np.pi if cov is not None else np.array([np.inf]*3)

            pos_arr.append(pos)
            rot_arr.append(rot)

        ci_pos = 1.96 * np.array(pos_arr)
        ci_rot = 1.96 * np.array(rot_arr)

        pos_labels = ['x', 'y', 'z']
        rot_labels = ['Roll', 'Pitch', 'Yaw']

        pos_error = self.vector_estim()[:, 3:]
        rot_error = self.vector_estim()[:, :3]
        gt = Geometry.SE3.as_vector(self.Trc, radians=False)
        for i in range(3):
            pos_ax = conf_pos[i]
            rot_ax = conf_rot[i]

            pos = pos_error[:, i]
            pos_ax.plot(t, pos, label=f'Estimated {pos_labels[i]}')
            pos_ax.fill_between(t, pos - ci_pos[:, i], pos + ci_pos[:, i], alpha=0.2, label='95% CI')
            pos_ax.axhline(gt[i + 3], color='red', linestyle='--', label='Ground truth')
            pos_ax.set_ylabel('[m]')
            pos_ax.set_xlabel(xlabel)
            pos_ax.legend()
            pos_ax.grid()
            pos_ax.set_title(pos_labels[i])
            pos_ax.set_ylim([gt[i + 3] - self.max_ylim_pos, gt[i + 3] + self.max_ylim_pos])

            rot = rot_error[:, i]
            rot_ax.plot(t, rot, label=f'Estimated {rot_labels[i]}')
            rot_ax.fill_between(t, rot - ci_rot[:, i], rot + ci_rot[:, i], alpha=0.2, label='95% CI')
            rot_ax.axhline(gt[i], color='red', linestyle='--', label='Ground truth')
            rot_ax.set_ylabel('[deg]')
            rot_ax.set_xlabel(xlabel)
            rot_ax.legend()
            rot_ax.grid()
            rot_ax.set_title(rot_labels[i])
            rot_ax.set_ylim([gt[i] - self.max_ylim_rot, gt[i] + self.max_ylim_rot])

    def plot_conf(self, t, conf_pos, conf_rot, xlabel) -> None:
        covs = self.covs
        Trc_estims = self.Trc_results
        pos_arr = []
        rot_arr = []
        for (cov, Trc_estim) in zip(covs, Trc_estims):

            if cov is not None: cov = Geometry.SE3.transform_cov(Trc_estim, cov)

            pos = np.sqrt(np.diag(cov[3:, 3:])) if cov is not None else np.array([-1]*3)
            rot = np.sqrt(np.diag(cov[:3, :3])) * 180 / np.pi if cov is not None else np.array([-1]*3)

            pos_arr.append(pos)
            rot_arr.append(rot)

        ci_pos = 1.96 * np.array(pos_arr)
        ci_rot = 1.96 * np.array(rot_arr)

        pos_labels = ['x', 'y', 'z']
        rot_labels = ['Rot X (~roll)', 'Rot Y (~pitch)', 'Rot Z (~yaw)']

        pos_error = self.se3_error()[:, 3:]
        rot_error = self.se3_error()[:, :3] * 180 / np.pi

        for i in range(3):
            pos_ax = conf_pos[i]
            rot_ax = conf_rot[i]

            pos_ax.plot(t, np.abs(pos_error[:, i]), label=f'Error {pos_labels[i]}')
            pos_ax.plot(t, ci_pos[:, i], label='95% CI')
            pos_ax.set_ylabel('[m]')
            pos_ax.set_xlabel(xlabel)
            pos_ax.legend()
            pos_ax.grid()
            pos_ax.set_title(pos_labels[i])
            # pos_ax_ylim = pos_ax.get_ylim()[1]
            # if pos_ax_ylim > self.max_ylim:
            #     pos_ax_ylim = self.max_ylim
            # pos_ax.set_ylim([0, pos_ax_ylim])

            rot_ax.plot(t, np.abs(rot_error[:, i]), label=f'Error {rot_labels[i]}')
            rot_ax.plot(t, ci_rot[:, i], label='95% CI')
            rot_ax.set_ylabel('[deg]')
            rot_ax.set_xlabel(xlabel)
            rot_ax.legend()
            rot_ax.grid()
            rot_ax.set_title(rot_labels[i])
            # rot_ax_ylim = rot_ax.get_ylim()[1]
            # if rot_ax_ylim > self.max_ylim:
            #     rot_ax_ylim = self.max_ylim
            # rot_ax.set_ylim([0, rot_ax_ylim])


    def show(self, xlabel):
        _, ax1 = plt.subplots(2, 3, tight_layout=True)
        _, ax2 = plt.subplots(2, 3, tight_layout=True)
        conf_pos, conf_rot = ax1[0, :], ax1[1, :]
        conf_pos2, conf_rot2 = ax2[0, :], ax2[1, :]

        t = self.t
        self.plot_conf(t, conf_pos, conf_rot, xlabel)
        self.plot_conf2(t, conf_pos2, conf_rot2, xlabel)
        
        plt.show()






class PlotVisualization:

    @staticmethod
    def apply_default_settings(ax: plt.Axes) -> plt.Axes:
        ax.grid()
        ax.legend()
        return ax

    @staticmethod
    def __get_poses(values: gtsam.Values) -> list[Pose3]:
        poses = gtsam.utilities.allPose3s(values)
        keys = poses.keys()
        result = []
        for key in keys:
            if chr(Symbol(key).chr()) != 'x': continue
            result.append(poses.atPose3(key))
            
        return result
    
    @classmethod
    def __get_logmap(clc, values: gtsam.Values) -> list[np.ndarray[6]]:
        return  np.array([Pose3.Logmap(pose) for pose in clc.__get_poses(values)])

    @classmethod
    def __plot_angle(clc, time: np.ndarray[1], values: gtsam.Values, dim = 0, label = '', ax: plt.Axes = None) -> plt.Axes:
        if not ax: fig, ax = plt.subplots()
        trajectory = clc.__get_logmap(values)
        att = trajectory[:, :3] * 180/np.pi
        ax.plot(time, att[:, dim], label=label)
        return ax
    
    @classmethod
    def __plot_position(clc, time: np.ndarray[1], values: gtsam.Values, dim = 0, label = '', ax: plt.Axes = None) -> plt.Axes:
        if not ax: fig, ax = plt.subplots()
        trajectory = clc.__get_logmap(values)
        pos = trajectory[:, 3:]
        ax.plot(time, pos[:, dim], label=label)
        return ax

    @classmethod
    def trajectory_roll(clc, time: np.ndarray[1], values: gtsam.Values, ax: plt.Axes = None) -> plt.Axes:
        return clc.__plot_angle(time, values, dim=0, label='roll', ax=ax)

    @classmethod
    def trajectory_pitch(clc, time: np.ndarray[1], values: gtsam.Values, ax: plt.Axes = None) -> plt.Axes:
        return clc.__plot_angle(time, values, dim=1, label='pitch', ax=ax)

    @classmethod
    def trajectory_yaw(clc, time: np.ndarray[1], values: gtsam.Values, ax: plt.Axes = None) -> plt.Axes:
        return clc.__plot_angle(time, values, dim=2, label='yaw', ax=ax)

    @classmethod
    def trajectory_x(clc, time: np.ndarray[1], values: gtsam.Values, ax: plt.Axes = None) -> plt.Axes:
        return clc.__plot_position(time, values, dim=0, label='x', ax=ax)
    
    @classmethod
    def trajectory_y(clc, time: np.ndarray[1], values: gtsam.Values, ax: plt.Axes = None) -> plt.Axes:
        return clc.__plot_position(time, values, dim=1, label='y', ax=ax)
    
    @classmethod
    def trajectory_z(clc, time: np.ndarray[1], values: gtsam.Values, ax: plt.Axes = None) -> plt.Axes:
        return clc.__plot_position(time, values, dim=2, label='z', ax=ax)


def get_covariance_ellipsoid(pose: Pose3 | Pose2, cov, n_std: float = 2.0) -> np.ndarray:
    # Generate unit sphere
    num_points = 30
    u = np.linspace(0, 2 * np.pi, num_points)
    v = np.linspace(0, np.pi, num_points)
    
    origin = pose.translation()
    n_dim = len(origin)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    radii = n_std * np.sqrt(eigenvalues)


    if n_dim == 2:
        x = np.cos(u)
        y = np.sin(u)
        ellipsoid = np.array([radii[0] * x, radii[1] * y])

    if n_dim == 3:
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))
        ellipsoid = np.array([radii[0] * x, radii[1] * y, radii[2] * z])

    # Apply rotation (eigenvectors)
    ellipsoid = np.tensordot(eigenvectors, ellipsoid.reshape(n_dim, -1), axes=1).reshape(n_dim, *x.shape)


    # Translate to the mean
    for i in range(n_dim): ellipsoid[i] += origin[i]

    return ellipsoid
    

def draw_covariance2D(ax, pose, cov) -> None:
    # https://docs.ros.org/en/kinetic/api/gtsam/html/plot_8py_source.html
    # https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html

    cov = cov[:2, :2] # 2D covariance
    ellipsoid = get_covariance_ellipsoid(pose, cov)
    ax.plot(ellipsoid[0], ellipsoid[1], color='b')

def draw_covariance3D(ax, pose, cov) -> None:
    # https://docs.ros.org/en/kinetic/api/gtsam/html/plot_8py_source.html
    # https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html

    cov = cov[:3, :3] # 3D covariance
    ellipsoid = get_covariance_ellipsoid(pose, cov)

    ax.plot_wireframe(ellipsoid[0], ellipsoid[1], ellipsoid[2], color='b')
   
def plot_graph2D(graph: gtsam.NonlinearFactorGraph, result: gtsam.Values, draw_cov=True, ax = None) -> None:

    if not ax: fig, ax = plt.subplots()

    marginals = gtsam.Marginals(graph, result)
    poses = gtsam.utilities.allPose2s(result)
    keys = poses.keys()
    origins = []
    for key in keys:
        pose = poses.atPose2(key)
        origin = pose.translation()
        cov = marginals.marginalCovariance(key)
        if draw_cov: draw_covariance2D(ax, pose, cov)

        origins.append(origin)

    origins = np.array(origins)
    ax.plot(origins[:, 0], origins[:, 1], '-o')


def plot_graph3D(graph: gtsam.NonlinearFactorGraph, result: gtsam.Values, draw_cov=True, ax = None) -> None:

    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

    if draw_cov: marginals = gtsam.Marginals(graph, result)
    poses = gtsam.utilities.allPose3s(result)
    keys = poses.keys()
    origins = []
    for key in keys:
        pose = poses.atPose3(key)
        origin = pose.translation()
        if draw_cov:
            cov = marginals.marginalCovariance(key)
            draw_covariance3D(ax, pose, cov)

        origins.append(origin)

    origins = np.array(origins)
    ax.plot(origins[:, 0], origins[:, 1], origins[:, 2], '-o')
