

from src.backend import Optimizer
from src.structs import Camera
from src.util import Geometry
import json

from src.visualization import GraphVisualization

class BackendRundown:
    
    def __init__(self) -> None:
        self.optimizer = Optimizer()
        self.load_map_points('./output/MapPoints.json')

        self.camera = Camera([458.654, 457.296, 367.215, 248.375], [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]) # More logic around camera should be added
        T = Geometry.SE3()
        self.optimizer.add_extrinsic_node(T, 0)

    def load_map_points(self, path):
        with open(path, 'r') as f: self.map_points = json.load(f)


    def start(self):
        pass

if __name__ == '__main__':

    app = BackendRundown()
    app.start()

    GraphVisualization.FactorGraphVisualization.draw_factor_graph('./output/', app.optimizer.graph, app.optimizer.current_estimate)