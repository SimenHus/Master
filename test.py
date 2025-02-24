import argparse

import matplotlib.pyplot as plt
import numpy as np

import Simulation
import Simulation.PlanarLandmark3D
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

    steps = 10
    sim = Simulation.PlanarLandmarkAltFactor(steps)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    origins = np.array([(node).translation() for node in sim.trajectory.trajectory])

    sim.simulate_all()
    ax.grid()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_zlim([-5, 5])

    plot_graph3D(sim.graph, sim.current_estimate, ax=ax, draw_cov=False)
    ax.plot(origins[:, 0], origins[:, 1], origins[:, 2], '-o')

    FactorGraphVisualization.draw_factor_graph(OUTPUT_FOLDER, sim.graph, sim.current_estimate)

    # for i in range(steps):
    #     sim.simulate_step()
    #     key = i + sim.key_start
    #     ax.cla()
    #     ax.grid()
    #     ax.set_xlabel('x')
    #     ax.set_ylabel('y')
    #     ax.set_zlabel('z')
    #     ax.set_zlim([-5, 5])
    #     plot_graph3D(sim.graph, sim.current_estimate, ax=ax, draw_cov=False)
    #     ax.plot(origins[:key + 1, 0], origins[:key + 1, 1], origins[:key + 1, 2], '-o')
    #     plt.pause(1)

    print(f'True extrinsics: {sim.camera_extrinsics}')
    print(f'Estimated extrinsics: {sim.current_estimate.atPose3(0)}')
    # print(f'Estimated cov: {sim.camera_extrinsics_covariance}')
    print(f'True landmarks: {sim.landmarks}')
    print(f'Estimated landmarks: {[sim.current_estimate.atPoint3(sim.landmark_index + i) for i in range(len(sim.landmarks))]}')

    plt.show()



if __name__ == '__main__':
    main()