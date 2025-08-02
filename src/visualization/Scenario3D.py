

import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt

from src.backend import NodeType

class Scenario3D:
    def __init__(self, result):

        fig = plt.figure()
        self.ax = fig.add_subplot(projection='3d')

        gtsam_plot.plot_3d_points(fig.number, result, 'rx')

        i = 0
        while result.exists(NodeType.REFERENCE(i)):
            pose_i = result.atPose3(NodeType.REFERENCE(i))
            gtsam_plot.plot_pose3(fig.number, pose_i, 10)
            i += 1
