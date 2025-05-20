
from gtsam import ISAM2Params, ISAM2
from gtsam import NonlinearFactorGraph, Values, Symbol
from gtsam import SmartProjectionPoseFactorCal3_S2, SmartProjectionParams, BetweenFactorPose3
from gtsam import NonlinearEqualityPose3, Cal3_S2
from gtsam import noiseModel
from gtsam.symbol_shorthand import X, T, C

from .factors import BetweenFactorCamera

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.util import Geometry
    from src.structs import Camera

class Optimizer:
    def __init__(self) -> None:
        self.new_factors = NonlinearFactorGraph()
        self.new_nodes = Values()
        self.smart_factors: dict[int: SmartProjectionPoseFactorCal3_S2] = {}

        parameters = ISAM2Params()
        parameters.setRelinearizeThreshold(0.1)
        parameters.relinearizeSkip = 1
        self.isam = ISAM2(parameters)

    def add_extrinsic_node(self, Twc_init: 'Geometry.SE3', camera_id: int) -> None:
        self.new_nodes.insert(T(camera_id), Twc_init)

    def add_camera_node(self, Twc_init: 'Geometry.SE3', camera_pose_id: int) -> None:
        self.new_nodes.insert(C(camera_pose_id), Twc_init)

    def add_ref_node(self, value: 'Geometry.SE3', ref_index: int) -> None:
        self.new_nodes.insert(X(ref_index), value)

    def add_ref_equality(self, value: 'Geometry.SE3', ref_index: int) -> None:
        self.new_factors.add(NonlinearEqualityPose3(X(ref_index), value))

    def add_camera_between_factor(self, ref_index: int, camera_id: int, camera_pose_id: int, sigmas: list) -> None:
        noise_model = noiseModel.Diagonal.Sigmas(sigmas)
        self.new_factors.add(BetweenFactorCamera(X(ref_index), T(camera_id), C(camera_pose_id), noise_model))

    def add_ref_odom_factor(self, from_index: int, to_index: int, odom: 'Geometry.SE3', sigmas: list) -> None:
        noise_model = noiseModel.Diagonal.Sigmas(sigmas)
        self.new_factors.add(BetweenFactorPose3(X(from_index), X(to_index), odom, noise_model))
    
    def add_camera_odom_factor(self, from_index: int, to_index: int, odom: 'Geometry.SE3', sigmas: list) -> None:
        noise_model = noiseModel.Diagonal.Sigmas(sigmas)
        self.new_factors.add(BetweenFactorPose3(C(from_index), C(to_index), odom, noise_model))

    def add_projection_factor(self, map_point_id: int, pixels: tuple, pose_id: int, camera: 'Camera') -> None:
        smart_params = SmartProjectionParams()
        pixel_noise = noiseModel.Isotropic.Sigma(2, 1.0)
        K = Cal3_S2(camera.parameters_with_skew)
        if map_point_id not in self.smart_factors.keys():
            self.smart_factors[map_point_id] = SmartProjectionPoseFactorCal3_S2(pixel_noise, K, smart_params)
            self.new_factors.add(self.smart_factors[map_point_id])
        self.smart_factors[map_point_id].add(pixels, C(pose_id))

    def add_camera_prior(self, value: 'Geometry.SE3', node_id: int, sigma: list) -> None:
        noise_model = noiseModel.Diagonal.Sigmas(sigma)
        self.new_factors.addPriorPose3(C(node_id), value, noise_model)

    def check_node_exists(self, node_id: Symbol) -> bool:
        in_isam = self.isam.valueExists(node_id)
        in_new_nodes = self.new_nodes.exists(node_id)
        return in_isam or in_new_nodes


    def optimize(self, extra_updates: int = 0) -> None:
        # Perform incremental update to iSAM
        self.isam.update(self.new_factors, self.new_nodes)
        self.new_nodes.clear()
        self.new_factors.resize(0)
        for _ in range(extra_updates): self.isam.update()

    def get_extrinsic_node_estimate(self, node_id: int) -> 'Geometry.SE3':
        node_symb = T(node_id)
        # if not self.isam.valueExists(node_symb): return
        return self.current_estimate.atPose3(node_symb)
    
    def get_ref_node_estimate(self, node_id: int) -> 'Geometry.SE3':
        node_symb = X(node_id)
        return self.current_estimate.atPose3(node_symb)
    
    def get_camera_node_estimate(self, node_id: int) -> 'Geometry.SE3':
        node_symb = C(node_id)
        return self.current_estimate.atPose3(node_symb)

    def get_visualization_variables(self) -> tuple[NonlinearFactorGraph, Values]:
        return self.isam.getFactorsUnsafe(), self.current_estimate

    @property
    def current_estimate(self) -> Values:
        return self.isam.calculateBestEstimate()