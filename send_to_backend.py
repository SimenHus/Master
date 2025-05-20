

from src.backend import Optimizer
from src.structs import Camera
from src.util import Geometry
import json

from src.visualization import GraphVisualization

class BackendRundown:
    
    def __init__(self) -> None:
        self.optimizer = Optimizer()
        self.load_map_points('./output/MapPoints.json')
        self.load_keyframes('./output/KeyFrames.json')

        self.camera = Camera([458.654, 457.296, 367.215, 248.375], [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]) # More logic around camera should be added

    def load_map_points(self, path):
        with open(path, 'r') as f: self.map_points = json.load(f)

    def load_keyframes(self, path):
        with open(path, 'r') as f: self.keyframes = json.load(f)


    def start(self):
        T0 = Geometry.SE3()
        # self.optimizer.add_extrinsic_node(Twc_init, 0)
        self.optimizer.add_camera_prior(T0, 0, [1e3]*6)
        odom = Geometry.SE3.Expmap([0, 0, 0, 1, 0, 0])
        prev_pose = T0.compose(odom.inverse())
        for i, (kf_id, keyframe) in enumerate(self.keyframes.items()):
            # pose = Geometry.SE3(keyframe['Twc'])
            pose = prev_pose.compose(odom)
            kf_id_int = int(keyframe['id'])
            self.optimizer.add_camera_node(pose, kf_id_int)
            for map_point in self.map_points.values():
                # if len(map_point['observations'].values()) < 3: continue # Skip mappoints with few observations
                if kf_id not in map_point['observations'].keys(): continue
                mp_id = int(map_point['id'])
                index = map_point['observations'][kf_id]
                kp = keyframe['keypoints'][index]
                self.optimizer.add_projection_factor(mp_id, kp, kf_id_int, self.camera)
            if i > 0:
                self.optimizer.add_camera_odom_factor(i-1, i, Geometry.SE3.between(prev_pose, pose), [1e0]*6)
                self.optimizer.optimize()
            prev_pose = pose


if __name__ == '__main__':

    app = BackendRundown()
    app.start()
    graph, estimate = app.optimizer.get_visualization_variables()
    GraphVisualization.FactorGraphVisualization.draw_factor_graph('./output/', graph, estimate)