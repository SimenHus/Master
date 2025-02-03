import argparse

import matplotlib.pyplot as plt
import numpy as np

from Simulation import Simulations
from Visualization.PlotVisualization import plot_graph2D
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
    sim = Simulations.Planar2D(steps)

    fig = plt.figure()
    ax = fig.add_subplot()
    origins = np.array([node.translation() for node in sim.trajectory.trajectory])

    sim.simulate_all()
    ax.grid()
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plot_graph2D(sim.graph, sim.current_estimate, ax=ax, draw_cov=True)
    ax.plot(origins[:, 0], origins[:, 1], '-o')

    FactorGraphVisualization.draw_factor_graph(OUTPUT_FOLDER, sim.graph, sim.current_estimate)

    plt.show()



if __name__ == '__main__':
    main()