
from gtsam import ISAM2Params, ISAM2
from gtsam import NonlinearFactorGraph, Values, Symbol
from gtsam import SmartProjectionPoseFactorCal3_S2, SmartProjectionParams, SmartProjectionPoseFactorCal3DS2
from gtsam import Cal3_S2, Cal3DS2
from gtsam import DegeneracyMode, LinearizationMode
from gtsam import BetweenFactorPose3
from gtsam import NonlinearEqualityPose3
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
        self.smart_params = SmartProjectionParams()
        self.smart_params.setDegeneracyMode(DegeneracyMode.ZERO_ON_DEGENERACY)
        self.smart_params.setLinearizationMode(LinearizationMode.HESSIAN)
        self.completed_factors = []

        # Noise model: https://gtsam.org/2019/09/20/robust-noise-model.html
        # Does not work with smartfactors???
        # self.pixel_noise = noiseModel.Robust.Create(
        #     noiseModel.mEstimator.Huber.Create(1.345),
        #     noiseModel.Isotropic.Sigma(2, 1.0)
        # )
        self.pixel_noise = noiseModel.Isotropic.Sigma(2, 1.0)

        parameters = ISAM2Params()
        # parameters.setRelinearizeThreshold(0.1)
        # parameters.relinearizeSkip = 1
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
        extrinsic_noise = noiseModel.Robust.Create(
            noiseModel.mEstimator.Huber.Create(1.345),
            noise_model
        )
        self.new_factors.add(BetweenFactorCamera(X(ref_index), T(camera_id), C(camera_pose_id), extrinsic_noise))

    def add_ref_odom_factor(self, from_index: int, to_index: int, odom: 'Geometry.SE3', sigmas: list) -> None:
        noise_model = noiseModel.Diagonal.Sigmas(sigmas)
        self.new_factors.add(BetweenFactorPose3(X(from_index), X(to_index), odom, noise_model))
    
    def add_camera_odom_factor(self, from_index: int, to_index: int, odom: 'Geometry.SE3', sigmas: list) -> None:
        noise_model = noiseModel.Diagonal.Sigmas(sigmas)
        self.new_factors.add(BetweenFactorPose3(C(from_index), C(to_index), odom, noise_model))

    def update_projection_factor(self, map_point_id: int, normed_pixels: tuple, pose_id: int, camera: 'Camera') -> None:
        if map_point_id not in self.smart_factors.keys():
            K = Cal3_S2(camera.parameters_with_skew)
            self.smart_factors[map_point_id] = SmartProjectionPoseFactorCal3_S2(self.pixel_noise, K, self.smart_params)

        pixels = camera.normed_to_pixels(normed_pixels)
        self.smart_factors[map_point_id].add(pixels, C(pose_id))

    def mark_complete_projection_factor(self, map_point_id: int) -> None:
        if map_point_id in self.completed_factors: return # Return if already completed
        self.completed_factors.append(map_point_id) # Mark as complete
        self.new_factors.add(self.smart_factors[map_point_id]) # Add factor to graph and make it immutable to further changes

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
        if not self.isam.valueExists(node_symb): return self.new_nodes.atPose3(node_symb)
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