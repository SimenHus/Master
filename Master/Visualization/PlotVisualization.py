
import gtsam # https://gtbook.github.io/gtsam-examples/intro.html
from gtsam import Point3, Point2, Pose3, Pose2, Rot3, Rot2
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d

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

    marginals = gtsam.Marginals(graph, result)
    poses = gtsam.utilities.allPose3s(result)
    keys = poses.keys()
    origins = []
    for key in keys:
        pose = poses.atPose3(key)
        origin = pose.translation()
        cov = marginals.marginalCovariance(key)
        if draw_cov: draw_covariance3D(ax, pose, cov)

        origins.append(origin)

    origins = np.array(origins)
    ax.plot(origins[:, 0], origins[:, 1], origins[:, 2], '-o')
