
from src.backend import BAOptimizer, NodeType
from src.structs import Camera
from src.util import Geometry, DataLoader, Time
from src.util.ProgressBar import ProgressBar
from src.visualization import Visualization, LiveTrajectory3D, Scenario3D
from src.simulation import TrajectoryGenerator, SeaStates, LandmarkGenerator

import numpy as np

import matplotlib.pyplot as plt

import time



class Application:

    def __init__(self, w_list, settings=SeaStates()) -> None:

        self.traj_settings = settings
        self.mps = LandmarkGenerator.grid_mps()

        self.w_list = w_list

        self.camera, _ = DataLoader.load_stx_camera()
        self.Trc = Geometry.SE3.from_vector(np.array([-5.5, 7.4, -87.9, 30.14, -10.32, -15.46]), radians=False)
        self.Trc = Geometry.SE3.NED_to_RDF(self.Trc)

        # Trc_noise = Geometry.SE3.from_vector(np.array([-10, 14, -23., 3.2, -4.25, -3.15]), radians=False)
        # self.Trc_init = self.Trc.compose(Trc_noise)
        self.Trc_init = Geometry.SE3.from_vector(np.array([0, 0, -90., 0, 0, 0]))
        self.Trc_init = Geometry.SE3.NED_to_RDF(self.Trc_init)

        self.Trc_prior_sigmas = np.array([1.]*6)*1e6

    def landmark_noise(self) -> Geometry.Point3:
        magnitude = 4
        return np.random.rand(3) * magnitude

    def optimize_traj(self, optimizer: BAOptimizer, traj: list[Geometry.SE3]) -> None:
         
        # Add initial guess and prior of extrinsics
        optimizer.add_node(self.Trc_init, self.camera.id, NodeType.EXTRINSIC)
        optimizer.add_prior(self.Trc_init, self.camera.id, NodeType.EXTRINSIC, self.Trc_prior_sigmas)


        # Add all landmarks to fg
        for j, point in enumerate(self.mps):
            optimizer.add_landmark(j, point + self.landmark_noise())

        # Loop through trajectory
        for i, Twr in enumerate(traj):

            optimizer.add_node(Twr, i, NodeType.REFERENCE) # Add current pose to fg
            optimizer.add_prior(Twr, i, NodeType.REFERENCE, np.array([1e-3]*6))

            Twc = Twr.compose(self.Trc) # Get camera in world
            for j, point in enumerate(self.mps):
                point_cam = Twc.transformTo(point) # Get point in camera frame
                pixels = self.camera.project(point_cam) # Project to pixels
                if np.any(pixels < 0) or np.any(pixels > 2056): continue # Skip map point if not observed in camera
        
                optimizer.add_landmark_observation(j, pixels, i, self.camera)


    def start(self):
        
        iterations = len(self.w_list)
        pb = ProgressBar(iterations)
        print(f'Simulating {iterations} trajectories...')
        pb.print(0)
        self.results = []
        self.Trc_results = []
        self.Trc_covs = []
        for i, w in enumerate(self.w_list):
            traj = TrajectoryGenerator.semi_circular(w=w, settings=self.traj_settings)
            current_optimizer = BAOptimizer()
            self.optimize_traj(current_optimizer, traj)

            result = current_optimizer.optimize()
            marginals = current_optimizer.marginals()

            Trc_cov = marginals.marginalCovariance(NodeType.EXTRINSIC(self.camera.id))
            Trc_estim = result.atPose3(NodeType.EXTRINSIC(self.camera.id))

            Trc_cov = Geometry.SE3.RDF_to_NED_cov(Trc_cov)
            Trc_estim = Geometry.SE3.RDF_to_NED(Trc_estim)
            

            self.results.append(result)
            self.Trc_results.append(Trc_estim)
            self.Trc_covs.append(Trc_cov)

            del current_optimizer
            pb.print(i+1)
        self.Trc = Geometry.SE3.RDF_to_NED(self.Trc)


    def show(self) -> None:

        # scenario = self.results[-1]
        # Scenario3D(scenario)

        xlabel = 'w'
        plots = Visualization.ReportPlot(self.w_list, self.Trc, self.Trc_results, self.Trc_covs)
        plots.show(xlabel)

            
if __name__ == '__main__':
    settings = SeaStates.preset_calm()
    settings = SeaStates.preset_moderate()
    settings = SeaStates.preset_rough()
    settings.set_steps(200)
    settings.set_distance(200)
    w_start = 0.
    w_end = 1.
    w_resolution = 100
    w_list = np.linspace(w_start, w_end, w_resolution)
    start = time.time()
    app = Application(w_list, settings=settings)
    app.start()
    print(f'Time elapsed: {time.time() - start}s')
    app.show()