

import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt

class LiveTrajectory3D:
    def __init__(self, node_type, delay=0.1, fignum=0):
        self.node_type = node_type
        self.delay = delay
        
        self.fignum = fignum
        plt.ion()
        fig = plt.figure(self.fignum)
        self.ax = fig.add_subplot(projection='3d')


    def update(self, result) -> None:
        plt.cla()
        gtsam_plot.plot_3d_points(self.fignum, result, 'rx')

        i = 0
        while result.exists(self.node_type(i)):
            pose_i = result.atPose3(self.node_type(i))
            gtsam_plot.plot_pose3(self.fignum, pose_i, 10)
            i += 1

        plt.pause(self.delay)

    def finished(self) -> None:
        plt.ioff()
        plt.show()