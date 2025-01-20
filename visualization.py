
from common import *


def draw_covariance2D(ax, pose, cov) -> None:
    # https://docs.ros.org/en/kinetic/api/gtsam/html/plot_8py_source.html
    # https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html

    origin = pose.translation() # Translation of ellipse
    cov = cov[0:2, 0:2] # 2D covariance
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    n_std = 2.0 # 95 % confidence interval
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]) # Ellipse angle
    width = 2*np.sqrt(eigenvalues[0])*n_std # 2*sqrt(eig_x)*standard deviations
    height = 2*np.sqrt(eigenvalues[1])*n_std # 2*sqrt(eig_y)*standard deviations
    e1 = patches.Ellipse(origin, width=width, height=height, angle=np.rad2deg(angle), fill=False)
    ax.add_patch(e1) # Add ellipse to ax

   
def plot_graph(graph: gtsam.NonlinearFactorGraph, result: gtsam.Values, draw_cov=True, ax = None) -> None:

    if not ax: fig, ax = plt.subplots()

    marginals = gtsam.Marginals(graph, result)
    poses = gtsam.utilities.allPose2s(result)
    keys = poses.keys()
    origins = []
    for key in keys:
        pose = poses.atPose2(key)
        origin = pose.translation()
        cov = marginals.marginalCovariance(key)
        draw_covariance2D(ax, pose, cov)

        origins.append(origin)

    origins = np.array(origins)
    ax.plot(origins[:, 0], origins[:, 1], '-o')


    # ax.xlim([-0.5, 5])
    # ax.ylim([-2, 2])
    ax.grid()
