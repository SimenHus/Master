
from gtsam import ISAM2Params, ISAM2, LevenbergMarquardtOptimizer
from gtsam import NonlinearFactorGraph, Values, Symbol
from gtsam import SmartProjectionPoseFactorCal3_S2, SmartProjectionParams
from gtsam import Cal3_S2
from gtsam import DegeneracyMode, LinearizationMode
from gtsam import BetweenFactorPose3, PriorFactorPose3
from gtsam import NonlinearEqualityPose3
from gtsam import noiseModel
from gtsam.symbol_shorthand import X, T, C

from .factors import BetweenFactorCamera
from src.util import Geometry
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.structs import Camera

class FactorIndexDB:
    next_factor_index = 0

    def __init__(self) -> None:
        self.db = {}
        self.map_point_observations = {}

    def increment(self) -> None:
        FactorIndexDB.next_factor_index += 1

    def set_next_index(self, id: int) -> None:
        FactorIndexDB.next_factor_index = id

    def add(self, id: int) -> None:
        self.db[id] = FactorIndexDB.next_factor_index
        self.increment()

    def add_map_point(self, id: int, pixels: tuple, pose_id: int) -> None:
        self.add(id)
        self.add_map_point_observation(id, pixels, pose_id)

    def __getitem__(self, key: int):
        return self.db[key]
    
    def __len__(self) -> int:
        return len(self.db)
    
    def size(self) -> int:
        return self.__len__()
    
    def __contains__(self, key: int) -> bool:
        """Check if supplied map point / id exists in database"""
        return key in self.db
    
    def add_map_point_observation(self, id: int, pixels: tuple, pose_id: int) -> None:
        if id not in self.map_point_observations: self.map_point_observations[id] = []
        self.map_point_observations[id].append([pixels, pose_id])

    def observations(self, id: int) -> list[tuple, int]:
        return self.map_point_observations[id]


class Optimizer:
    def __init__(self) -> None:
        self.master_graph = NonlinearFactorGraph() # Full graph, used for batch optimization
        self.new_factors = NonlinearFactorGraph()
        self.new_nodes = Values()

        self.factor_db = FactorIndexDB()
        self.smart_params = SmartProjectionParams()
        # self.smart_params.setDegeneracyMode(DegeneracyMode.ZERO_ON_DEGENERACY)
        self.smart_params.setLinearizationMode(LinearizationMode.HESSIAN)
        self.factors_to_update = []

        # Noise model: https://gtsam.org/2019/09/20/robust-noise-model.html
        # Does not work with smartfactors???
        # self.pixel_noise = noiseModel.Robust.Create(
        #     noiseModel.mEstimator.Huber.Create(1.345),
        #     noiseModel.Isotropic.Sigma(2, 1.0)
        # )
        self.pixel_noise = noiseModel.Isotropic.Sigma(2, 1.0) # 1 pixel std deviation in u and v

        self.isam_parameters = ISAM2Params()
        # self.isam_parameters.setRelinearizeThreshold(0.0)
        # self.isam_parameters.relinearizeSkip = 6
        self.isam = ISAM2(self.isam_parameters)

    def add_factor(self, factor, id: 'int | None' = None) -> None:
        self.new_factors.add(factor) # Add factor to set of factors which should be added to FG
        if id is None: self.factor_db.increment() # Increment number of factors in DB
        else: self.factor_db.add(id) # Remember factor information and increment number of factors in DB

    def add_extrinsic_node(self, Twc_init: 'Geometry.SE3', camera_id: int) -> None:
        self.new_nodes.insert(T(camera_id), Twc_init)

    def add_camera_node(self, Twc_init: 'Geometry.SE3', camera_pose_id: int) -> None:
        self.new_nodes.insert(C(camera_pose_id), Twc_init)

    def add_ref_node(self, value: 'Geometry.SE3', ref_index: int) -> None:
        self.new_nodes.insert(X(ref_index), value)

    def add_ref_equality(self, value: 'Geometry.SE3', ref_index: int) -> None:
        self.add_factor(NonlinearEqualityPose3(X(ref_index), value))

    def add_camera_between_factor(self, ref_index: int, camera_id: int, camera_pose_id: int, sigmas: list) -> None:
        noise_model = noiseModel.Diagonal.Sigmas(sigmas)
        extrinsic_noise = noiseModel.Robust.Create(
            noiseModel.mEstimator.Huber.Create(1.345),
            noise_model
        )
        self.add_factor(BetweenFactorCamera(X(ref_index), T(camera_id), C(camera_pose_id), extrinsic_noise))

    def add_ref_odom_factor(self, from_index: int, to_index: int, odom: 'Geometry.SE3', sigmas: list) -> None:
        noise_model = noiseModel.Diagonal.Sigmas(sigmas)
        self.add_factor(BetweenFactorPose3(X(from_index), X(to_index), odom, noise_model))
    
    def add_camera_odom_factor(self, from_index: int, to_index: int, odom: 'Geometry.SE3', sigmas: list) -> None:
        noise_model = noiseModel.Diagonal.Sigmas(sigmas)
        self.add_factor(BetweenFactorPose3(C(from_index), C(to_index), odom, noise_model))

    def update_projection_factor(self, map_point_id: int, pixels: tuple, pose_id: int, camera: 'Camera', Twc = Geometry.SE3()) -> None:
        self.factor_db.add_map_point_observation(map_point_id, pixels, C(pose_id)) # Include pixel observation in factor DB

        observations = self.factor_db.observations(map_point_id) # Get observations from factor DB
        if len(observations) < 2: return # Need 2 or more observations before adding to FG
        K = Cal3_S2(camera.parameters_with_skew)
        factor = SmartProjectionPoseFactorCal3_S2(self.pixel_noise, K, Twc, self.smart_params)
        for (kp, pose_id) in observations: factor.add(kp, pose_id) # Add all observations to the factor
        
        # If factor already exists in factor graph, update it and remove old
        if map_point_id in self.factor_db:
            self.factors_to_update.append(self.factor_db[map_point_id])

        # Add factor to factor graph
        self.add_factor(factor, map_point_id)

    def add_camera_prior(self, value: 'Geometry.SE3', node_id: int, sigma: list) -> None:
        noise_model = noiseModel.Diagonal.Sigmas(sigma)
        self.add_factor(PriorFactorPose3(C(node_id), value, noise_model))

    def check_node_exists(self, node_id: Symbol) -> bool:
        in_isam = self.isam.valueExists(node_id)
        in_new_nodes = self.new_nodes.exists(node_id)
        return in_isam or in_new_nodes

    def optimize(self, extra_updates: int = 0) -> None:
        for i in range(self.new_factors.size()): self.master_graph.add(self.new_factors.at(i)) # Add new nodes to master graph
        for index in self.factors_to_update: self.master_graph.remove(index) # Remove old factors from master graph

        self.factors_to_update = [fac for fac in self.factors_to_update if self.isam.valueExists(fac)]

        # Perform incremental update to iSAM, and remove/readd nodes to be updated
        self.isam.update(self.new_factors, self.new_nodes, self.factors_to_update)
        for _ in range(extra_updates): self.isam.update() # Perform extra updates

        self.new_nodes.clear() # Clear list of new nodes, since they are now added
        self.new_factors.resize(0) # Clear list of new factors, since they are now added
        self.factors_to_update.clear() # Clear list of updated factors


    def get_extrinsic_node_estimate(self, node_id: int) -> 'Geometry.SE3':
        return self._get_SE3_node(T(node_id))
    
    def get_ref_node_estimate(self, node_id: int) -> 'Geometry.SE3':
        return self._get_SE3_node(X(node_id))
    
    def get_camera_node_estimate(self, node_id: int) -> 'Geometry.SE3':
        return self._get_SE3_node(C(node_id))

    def get_visualization_variables(self) -> tuple[NonlinearFactorGraph, Values]:
        return self.master_graph, self.current_estimate

    def _get_SE3_node(self, node_symb: Symbol) -> 'Geometry.SE3':
        if self.current_estimate.exists(node_symb): return self.current_estimate.atPose3(node_symb)
        if self.new_nodes.exists(node_symb): return self.new_nodes.atPose3(node_symb)
        return None
    
    def _get_SE3_node_BA(self, node_symb: Symbol) -> 'Geometry.SE3':
        if self.global_bundle_adjustment_estimate.exists(node_symb):
            return self.global_bundle_adjustment_estimate.atPose3(node_symb)
        return None


    def get_camera_node_bundle_solution(self, node_id: int) -> 'Geometry.SE3':
        return self._get_SE3_node_BA(C(node_id))

    @property
    def current_estimate(self) -> Values:
        return self.isam.calculateBestEstimate()

    @property    
    def global_bundle_adjustment_estimate(self) -> Values:
        return LevenbergMarquardtOptimizer(self.master_graph, self.current_estimate).optimize()