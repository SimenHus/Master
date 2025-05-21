

from src.backend import Optimizer
from src.structs import Camera
from src.util import Geometry, DataLoader
import json
import numpy as np

from src.visualization import Visualization

class BackendRundown:
    
    def __init__(self) -> None:
        self.optimizer = Optimizer()
        self.load_map_points('./output/MapPoints.json')
        self.load_keyframes('./output/KeyFrames.json')
        self.load_ref_pose_gt('./datasets/osl/data')

        self.camera, self.Twc_gt = DataLoader.load_stx_camera('./datasets/osl')

    def load_map_points(self, path):
        with open(path, 'r') as f: self.map_points = json.load(f)

    def load_keyframes(self, path):
        # with open(path, 'r') as f: self.keyframes = sorted(json.load(f), key=lambda x: x['id'])
        with open(path, 'r') as f: self.keyframes = json.load(f)

    def load_ref_pose_gt(self, path):
        poses = DataLoader.load_stx_poses(path)
        pose_dict = {pose.timestep: pose.pose for pose in poses}
        self.poses = pose_dict


    def start(self):
        for kf_id, keyframe in self.keyframes.items():
            t0 = keyframe['timestep']
            id0 = int(kf_id)
            break
        T0 = self.poses[t0]
        att_noisy = np.array([0, 0, 180]) * np.pi / 180
        att_noisy = Geometry.SO3.Logmap(self.Twc_gt.rotation()) + np.array([0.5, -0.5, -0.5])
        pos_noisy = self.Twc_gt.translation() + np.array([0.8, -0.2, 0.5])
        Twc_noisy = Geometry.SE3(Geometry.SO3.Expmap(att_noisy), pos_noisy)
        prior_sigmas = [0.5, 0.5, 0.5, 1, 1, 1]
        camera_between_sigmas = [0.02, 0.02, 0.02, 0.05, 0.05, 0.05]

        self.optimizer.add_extrinsic_node(Twc_noisy, self.camera.id) # Add initial guess for extrinsics
        self.optimizer.add_camera_prior(T0.compose(Twc_noisy), id0, prior_sigmas) # Add prior for first camera pose based on initial guess of extrinsics
        for i, (kf_id, keyframe) in enumerate(self.keyframes.items()):
            T_rel_est = self.optimizer.get_extrinsic_node_estimate(self.camera.id)
            ref_pose = self.poses[keyframe['timestep']] # Get pose from file
            # cam_pose = ref_pose.compose(self.Twc_gt)
            cam_pose = ref_pose.compose(T_rel_est) # Get estimated camera pose
            kf_id_int = int(keyframe['id'])
            self.optimizer.add_ref_node(ref_pose, kf_id_int) # Add node for reference pose
            self.optimizer.add_ref_equality(ref_pose, kf_id_int) # Add factor to constrain reference pose
            self.optimizer.add_camera_node(cam_pose, kf_id_int) # Add node for camera pose
            self.optimizer.add_camera_between_factor(kf_id_int, self.camera.id, kf_id_int, camera_between_sigmas) # Add factor to constrain camera to reference frame
            if i > 0:
                self.optimizer.optimize()
            for map_point in self.map_points.values():
                observations = map_point['observations']
                if len(observations) < 3: continue # Skip map points with few observations
                if kf_id not in observations.keys(): continue # Map point not in frame
                mp_id = int(map_point['id'])
                kp = keyframe['keypoints_und'][observations[kf_id]]
                self.optimizer.update_projection_factor(mp_id, kp, kf_id_int, self.camera)
                sorted_obs = sorted(observations, key=lambda x: int(x)) # Sort by KF IDs
                if sorted_obs.index(kf_id) > 2: # If observed more than three times, mark as complete
                    self.optimizer.mark_complete_projection_factor(mp_id)
                    self.optimizer.optimize()


if __name__ == '__main__':

    app = BackendRundown()
    app.start()
    graph, estimate = app.optimizer.get_visualization_variables()
    Visualization.FactorGraphVisualization.draw_factor_graph('./output/', graph, estimate)
    
    gt = Geometry.SE3.Logmap(app.Twc_gt)
    estim = Geometry.SE3.Logmap(app.optimizer.get_extrinsic_node_estimate(app.camera.id))
    print('Ground truth:')
    print(f'Position: {gt[3:]}')
    print(f'Rotation: {gt[:3]*180 / np.pi}')

    print('\n\nEstimate:')
    print(f'Position: {estim[3:]}')
    print(f'Rotation: {estim[:3]*180 / np.pi}') 