
from src.backend import Optimizer, NodeType
from src.structs import Camera
from src.util import Geometry, DataLoader, Time
from src.util.ProgressBar import ProgressBar
from src.visualization import Visualization, LiveTrajectory3D
from src.simulation import TrajectoryGenerator, SeaStates, LandmarkGenerator

import numpy as np

import matplotlib.pyplot as plt



class Application:

    def __init__(self, w_list, visualize=False, settings=SeaStates()) -> None:

        self.plot_3d = None
        if visualize: self.plot_3d = LiveTrajectory3D(NodeType.REFERENCE, delay=0.01)

        self.traj_settings = settings
        self.mps = LandmarkGenerator.grid_mps()

        self.w_list = w_list

        self.camera = Camera([50., 50., 50., 50.], [])

        self.Trc = Geometry.SE3.from_vector(np.array([-87.9, 7.4, -5.5, -0.14, 0.32, 5.46]), radians=False)
        Trc_noise = Geometry.SE3.from_vector(np.array([15, 7.5, -15.6, 5.1, 2.1, -3.1]), radians=False)
        self.Trc_init = self.Trc.compose(Trc_noise)
        self.Trc_prior_sigmas = np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1])*1e3

    def landmark_noise(self) -> Geometry.Point3:
        magnitude = 2
        return np.random.rand(3) * magnitude

    def optimize_traj(self, optimizer: Optimizer, traj: list[Geometry.SE3]) -> None:
         
        # Add initial guess and prior of extrinsics
        optimizer.add_node(self.Trc_init, self.camera.id, NodeType.EXTRINSIC)
        optimizer.add_prior(self.Trc_init, self.camera.id, NodeType.EXTRINSIC, self.Trc_prior_sigmas)


        # Add all landmarks to fg
        for j, point in enumerate(self.mps):
            optimizer.add_node(point + self.landmark_noise(), j, NodeType.LANDMARK)

        # Loop through trajectory
        for i, Twr in enumerate(traj):
            extra_optims = 0
            
            optimizer.add_node(Twr, i, NodeType.REFERENCE) # Add current pose to fg
            optimizer.add_pose_equality(Twr, i, NodeType.REFERENCE) # Set current pose to not be optimized

            Twc = Twr.compose(self.Trc) # Get camera in world
            for j, point in enumerate(self.mps):
                point_cam = Twc.transformTo(point) # Get point in camera frame
                pixels = self.camera.project(point_cam) # Project to pixels
                if np.any(pixels < 0): continue # Skip map point if not observed in camera
        
                optimizer.add_extrinsic_projection_factor(j, pixels, i, self.camera) # Add observation to fg

            try:
                if i > 0:
                    optimizer.optimize(extra_optims) # Optimize after at least two timesteps have passed
            except Exception as e:
                print(e)
                # break

            if self.plot_3d is not None: self.plot_3d.update(optimizer.current_estimate)

    def start(self):
        
        iterations = len(self.w_list)
        pb = ProgressBar(iterations)
        print(f'Simulating {iterations} trajectories...')
        pb.print(0)
        self.Trc_results = []
        for i, w in enumerate(self.w_list):
            traj = TrajectoryGenerator.semi_circular(w=w, settings=self.traj_settings)
            current_optimizer = Optimizer(relin_skip=1, relin_thres=0.1)
            self.optimize_traj(current_optimizer, traj)
            self.Trc_results.append(current_optimizer.get_node_estimate(self.camera.id, NodeType.EXTRINSIC))
            del current_optimizer
            pb.print(i+1)
        if self.plot_3d is not None: self.plot_3d.finished()


    def get_error_metrics(self) -> tuple[list[float], list[float]]:
        rot_error = []
        pos_error = []
        for Trc_estim in self.Trc_results:
            error = Geometry.SE3.localCoordinates(self.Trc, Trc_estim)
            rot_error.append(np.linalg.norm(error[:3]))
            pos_error.append(np.linalg.norm(error[3:]))
        return rot_error, pos_error

    def show(self) -> None:
        fig, axs = plt.subplots(1, 2)
        pos_ax, rot_ax = axs
        t = self.w_list
        rot_error, pos_error = self.get_error_metrics()

        pos_ax.set_title('Position error')
        pos_ax.set_xlabel('w')
        pos_ax.set_ylabel('[m]')
        pos_ax.plot(t, pos_error)
        pos_ax.grid()
        pos_ax.legend()
        
        rot_ax.set_title('Rotation error')
        rot_ax.set_xlabel('w')
        rot_ax.set_ylabel('[rad]')
        rot_ax.plot(t, rot_error)
        rot_ax.grid()
        rot_ax.legend()


        plt.show()

            
if __name__ == '__main__':
    # settings = SeaStates()
    settings = SeaStates.preset_calm()
    # settings = SeaStates.preset_sway()
    settings.set_steps(10)
    w_start = -0.5
    w_end = 1.0
    w_resolution = 50
    w_list = np.linspace(w_start, w_end, w_resolution)
    app = Application(w_list, visualize=False, settings=settings)
    app.start()
    app.show()