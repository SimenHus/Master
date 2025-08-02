
from gtsam import LevenbergMarquardtOptimizer, Marginals
from gtsam import ISAM2Params, ISAM2
from gtsam import NonlinearFactorGraph, Values, NonlinearFactor
from gtsam import BetweenFactorPose3, PriorFactorPose3
from gtsam import NonlinearEqualityPose3, PriorFactorPoint3
from gtsam import noiseModel
from gtsam.symbol_shorthand import X, T, C, V, L


from .factors import ExtrinsicProjectionFactor, ExtrinsicProjectionFactorNumeric
from src.util import Geometry
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.structs import Camera

class NodeType:
    CAMERA = C
    REFERENCE = X
    EXTRINSIC = T
    VELOCITY = V
    LANDMARK = L


def PriorFactorMap(node_type: NodeType) -> NonlinearFactor:
    match node_type:
        case NodeType.REFERENCE | NodeType.EXTRINSIC | NodeType.CAMERA:
            return PriorFactorPose3
        case NodeType.LANDMARK:
            return PriorFactorPoint3


class BAOptimizer:
    def __init__(self) -> None:
        self.nodes_master = Values()
        self.factors_master = NonlinearFactorGraph()

        self.landmarks = {}
        self.landmark_observations = {}

        self.smart_pixel_noise = noiseModel.Isotropic.Sigma(2, 1.0) # Pixel std deviation in u and v
        self.generic_pixel_noise = noiseModel.Robust.Create(
            noiseModel.mEstimator.Huber.Create(1.345),
            self.smart_pixel_noise
        )


    def _add_factor(self, factor) -> None:
        self.factors_master.add(factor)

    def add_landmark(self, id: int, pos: Geometry.Vector3) -> None:
        if id in self.landmarks: return
        self.landmarks[id] = pos
        self.landmark_observations[id] = []

    def add_node(self, value: 'Geometry', id: int, node_type: NodeType) -> None:
        self.nodes_master.insert(node_type(id), value)

    def add_prior(self, value, id: int, node_type: NodeType, sigma: list) -> None:
        noise_model = noiseModel.Diagonal.Sigmas(sigma)
        key = node_type(id)
        factor = PriorFactorMap(node_type)
        self._add_factor(factor(key, value, noise_model))

    def add_landmark_observation(self, id: int, pixels: tuple, pose_id: int, camera: 'Camera') -> None:
        observation = [pixels, pose_id, camera]
        self.landmark_observations[id].append(observation)

    def handle_reprojections(self) -> None:
        thresh = 2
        # Remove landmarks that are not observed often enough
        for landmark_id, observations in self.landmark_observations.items():
            if len(observations) < thresh: continue
            
            self.nodes_master.insert(NodeType.LANDMARK(landmark_id), self.landmarks[landmark_id])

            for (pixels, pose_id, camera) in observations:
                factor = ExtrinsicProjectionFactor(NodeType.REFERENCE(pose_id), NodeType.EXTRINSIC(camera.id), NodeType.LANDMARK(landmark_id),
                                                   camera, pixels, self.generic_pixel_noise)
                self.factors_master.add(factor)
        

    def optimize(self) -> Values:
        self.handle_reprojections()
        self.result = LevenbergMarquardtOptimizer(self.factors_master, self.nodes_master).optimize()
        return self.result
    
    def marginals(self) -> Marginals:
        return Marginals(self.factors_master, self.result)