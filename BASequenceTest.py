
from src.backend import BAOptimizer, NodeType, BAOptimizerGN
from src.util import Geometry, DataLoader, Time
from src.util.ProgressBar import ProgressBar
from src.visualization import Scenario3D, Visualization
from src.util.COLMAP import COLMAP

from pycolmap import Reconstruction

import numpy as np

class Application:
    
    def __init__(self, start_id, end_id, iterations, frame_skip, include_prev_iter) -> None:
        
        self.start_id = start_id
        self.end_id = end_id
        self.iterations = iterations
        self.include_prev_iter = include_prev_iter
        self.frame_skip = frame_skip

        print('Loading resources...')
        stx_data = DataLoader.load_stx_data()
        self.states: dict[int: Geometry.State] = {data.timestep: data.state for data in stx_data}

        self.camera, self.Trc = DataLoader.load_stx_camera()
        self.Trc = self.Trc.compose(Geometry.SE3.NED_to_RDF_map())

        Trc_init_1 = Geometry.SE3.from_vector(np.array([0, 0, 180, 0, 0, 0]), radians=False) # osl cam1 lens 0
        Trc_init_2 = Geometry.SE3.from_vector(np.array([0, 0, 90, 0, 0, 0]), radians=False) # osl cam1 lens 1, maybe 120 instead?
        Trc_init_3 = Geometry.SE3.from_vector(np.array([0, 0, 90, 0, 0, 0]), radians=False) # osl cam1 lens 2, maybe 60 instead?
        Trc_init_4 = Geometry.SE3.from_vector(np.array([0, 0, 90, 0, 0, 0]), radians=False) # asko therese cam1 lens 0

        inits = [Trc_init_1, Trc_init_2, Trc_init_3, Trc_init_4]
        Trc_init = inits[DataLoader.dataset_number() - 1]

        self.Trc_init = Geometry.SE3.NED_to_RDF(Trc_init)

        pose_dict = {timestep: state.pose for timestep, state in self.states.items()}
        self.colmap = COLMAP()
        self.colmap.filter_binned()
        self.colmap.align(pose_dict, self.Trc_init)
        self.colmap.filter_distance(pose_dict)
        # self.colmap.filter_spatial()
        print(f'Finished filtering with {len(self.colmap.landmarks)} landmarks left in the sequence')

        self.timesteps = sorted(self.colmap.poses.keys())

        print('Resources loaded')

    def optim_iteration(self, optimizer: BAOptimizer, active_window, Trc_init):
        rot_sigmas = np.array([1]*3)*1e6
        pos_sigmas = np.array([1]*3)*1e6
        extrinsic_prior_sigmas = np.append(rot_sigmas, pos_sigmas)

        optimizer.add_node(Trc_init, self.camera.id, NodeType.EXTRINSIC)
        optimizer.add_prior(Trc_init, self.camera.id, NodeType.EXTRINSIC, extrinsic_prior_sigmas)

        total_obs = 0
        for i, timestep in enumerate(active_window):
            
            state = self.states[timestep]
            Twr = state.pose

            optimizer.add_node(Twr, i, NodeType.REFERENCE)
            optimizer.add_prior(Twr, i, NodeType.REFERENCE, np.array([1e-3]*6))

            for (pixels, mp_id) in self.colmap.observations[timestep]:
                point3d = self.colmap.landmarks[mp_id]
                optimizer.add_landmark(mp_id, point3d)
                optimizer.add_landmark_observation(mp_id, pixels, i, self.camera)
                total_obs += 1

        print('\nPerforming bundle adjustment:')
        print(f' - Poses: {len(active_window)}')
        print(f' - Landmarks: {len(optimizer.landmarks)}')
        print(f' - Observations: {total_obs}')
            
        result = optimizer.optimize()
        print('Bundle adjustment finished')
        return optimizer.factors_master, result


    def start(self):
        start_id = self.start_id
        end_id = self.end_id
        iterations = self.iterations

        timesteps_active = self.timesteps[start_id:end_id]
        frames_per_iteration = len(timesteps_active) // iterations

        pb = ProgressBar(iterations)
        print(f'Processing {iterations} iterations...')
        pb.print(0)

        results = []
        estims = []
        covs = []
        self.t = []
        timer = Time.Timer()
        for i in range(iterations):
            timer.start()
            start = frames_per_iteration * i if not self.include_prev_iter else 0
            end = frames_per_iteration * (i + 1)
            if end > len(timesteps_active) or i == iterations -1: end = len(timesteps_active) + 1

            start_name = f'{start_id + 1 + start}'
            end_name = f'{start_id + end}'
            # self.t.append(f'{start_name}-{end_name}')
            print(f'\n\nIteration {i+1}, frames {start_name}-{end_name}')

            active_window = timesteps_active[start:end:self.frame_skip+1]

            optimizer = BAOptimizer()
            try:
                graph, result = self.optim_iteration(optimizer, active_window, self.Trc_init)
            except Exception as e:
                print(f'Failed to optimize: {e}')
                continue

            self.t.append(len(active_window))

            try:
                marginals = optimizer.marginals()
                Trc_cov = marginals.marginalCovariance(NodeType.EXTRINSIC(self.camera.id))
                Trc_cov = Geometry.SE3.RDF_to_NED_cov(Trc_cov)
            except Exception as e:
                print(f'Failed to calculate marginals: {e}')
                Trc_cov = None

            covs.append(Trc_cov)
            estim = result.atPose3(NodeType.EXTRINSIC(self.camera.id))
            estim = Geometry.SE3.RDF_to_NED(estim)
            estims.append(estim)
            results.append(result)
            del optimizer
            print(f'Iteration time: {timer.end()/60:.1f}m')
            pb.print(i+1)

        self.Trc = Geometry.SE3.RDF_to_NED(self.Trc)
        self.Trc_results = estims
        self.Trc_covs = covs
        self.results = results

    def show(self) -> None:

        scenario = self.results[-1]
        Scenario3D(scenario)
        print(f'GT: {Geometry.SE3.as_vector(self.Trc, radians=False)}')
        print(f'Last iteration result: {Geometry.SE3.as_vector(self.Trc_results[-1], radians=False)}')

        xlabel = 'Frames used'
        plots = Visualization.ReportPlot(self.t, self.Trc, self.Trc_results, self.Trc_covs)
        plots.show(xlabel)
            
            
if __name__ == '__main__':

    include_prev_iter = True
    iterations = 50
    start_id = 0
    end_id = -1
    frame_skip = 0

    app = Application(start_id, end_id, iterations, frame_skip, include_prev_iter)
    app.start()

    app.show()