import argparse

import matplotlib.pyplot as plt
import numpy as np

import Simulation
from Visualization.PlotVisualization import plot_graph3D
from Visualization.GraphVisualization import FactorGraphVisualization

import json

def load_config() -> None:
    with open('Config.json', 'r') as f: result = json.load(f)
    return result

def main() -> None:
    config = load_config()
    OUTPUT_FOLDER = config['dir']['output']

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dim', help='Choose 2D/3D simulation', default=3, type=int)
    # parser.add_argument()
    # args = parser.parse_args()
    # opts = args.opts
    # print(opts)

    steps = 30

    camera_prior = None
    camera_cov_prior = None
    for _ in range(3):
        sim = Simulation.Planar3DCamera(steps, camera_prior, camera_cov_prior)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        origins = np.array([(sim.camera_extrinsics * node).translation() for node in sim.trajectory.trajectory])

        sim.simulate_all()
        ax.grid()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_zlim([-5, 5])

        plot_graph3D(sim.graph, sim.current_estimate, ax=ax, draw_cov=False)
        ax.plot(origins[:, 0], origins[:, 1], origins[:, 2], '-o')

        FactorGraphVisualization.draw_factor_graph(OUTPUT_FOLDER, sim.graph, sim.current_estimate)

        camera_prior = sim.current_estimate.atPose3(0)
        # camera_cov_prior = sim.camera_extrinsics_covariance

        print(f'True extrinsics: {sim.camera_extrinsics}')
        print(f'Estimated extrinsics: {sim.current_estimate.atPose3(0)}')
        # print(f'Estimated cov: {sim.camera_extrinsics_covariance}')

    # plt.show()



if __name__ == '__main__':
    main()