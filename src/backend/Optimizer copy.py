
from gtsam import ISAM2Params, ISAM2, LevenbergMarquardtOptimizer
from gtsam import NonlinearFactorGraph, Values, Symbol, NonlinearFactor
from gtsam import SmartProjectionParams
from gtsam import SmartProjectionPoseFactorCal3_S2
# from gtsam import SmartProjectionPose3Factor
from gtsam import GenericProjectionFactorCal3_S2
from gtsam import Cal3_S2
from gtsam import DegeneracyMode, LinearizationMode
from gtsam import BetweenFactorPose3, PriorFactorPose3
from gtsam import NonlinearEqualityPose3, PriorFactorPoint3
from gtsam import noiseModel
from gtsam.symbol_shorthand import X, T, C, V, L
from gtsam import triangulatePoint3


import cv2
from dataclasses import dataclass
from .factors import BetweenFactorCamera, ReferenceAnchor, VelocityExtrinsicFactor, VelocityFactor, HandEyeFactor, ExtrinsicProjectionFactor
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

class Status:
    PENDING = 0
    ADDED = 1

def PriorFactorMap(node_type: NodeType) -> NonlinearFactor:
    match node_type:
        case NodeType.REFERENCE | NodeType.EXTRINSIC | NodeType.CAMERA:
            return PriorFactorPose3
        case NodeType.LANDMARK:
            return PriorFactorPoint3

@dataclass
class Factor:
    identifier: str
    status: Status
    factor: NonlinearFactor
    index: int = -1

    def __eq__(self, other) -> bool:
        if isinstance(other, Factor): return self.identifier == other.identifier
        if isinstance(other, str): return self.identifier == other
        return False

    def __hash__(self) -> int:
        return hash(self.identifier)
    

class FactorDatabase:
    def __init__(self) -> None:
        self.db: set[Factor] = set()
        self.map_point_observations: dict[str: list] = {} # Identifier: [pixels, pose_id]
        self.master_graph = NonlinearFactorGraph() # Factor graph containing all used factors (should be equal to isam fg)
        self.removal_indeces = []

    def add_pending(self, identifier: str, factor: NonlinearFactor) -> None:
        self.db.add(Factor(identifier, Status.PENDING, factor))
 
    def prepare_update(self) -> tuple[NonlinearFactorGraph, list]:
        next_index = self.master_graph.size() # Reset next index counter
        fg = NonlinearFactorGraph() # Prepare factor graph containing pending factors
        
        for fac in self.db:
            if fac.status != Status.PENDING: continue

            fg.add(fac.factor)
            self.master_graph.add(fac.factor)

            self.db.remove(fac) # Remove pending factor so that it can be updated
            fac.status = Status.ADDED # Change status
            fac.index = next_index # Set index in fg
            self.db.add(fac) # Re-add factor with updated values
            next_index += 1 # Increment fg index

        return fg, self.removal_indeces
    
    def add_map_point_observation(self, identifier: str, pixels: tuple, pose_id: int) -> bool:
        if identifier not in self.map_point_observations: self.map_point_observations[identifier] = []
        
        for obs in self.map_point_observations[identifier]:
            if obs[1] == pose_id: return False # Specific observation already added

        self.map_point_observations[identifier].append((pixels, pose_id))
        return True

    def remove(self, identifier: str) -> None:
        fac = self[identifier]
        if fac.status == Status.ADDED: self.removal_indeces.append(fac.index)
        self.db.remove(identifier)

    def finish_update(self) -> None:
        for index in self.removal_indeces: self.master_graph.remove(index) # Remove old factors from master graph
        self.removal_indeces.clear()

    def observations(self, identifier: str) -> list[tuple, int]:
        if identifier not in self.map_point_observations: return []
        return self.map_point_observations[identifier]
    
    def size(self) -> int:
        return self.__len__()
    
    def __contains__(self, identifier: str) -> bool:
        """Check if supplied map point / id exists in database"""
        return identifier in self.db

    def __getitem__(self, identifier: str) -> Factor:
        return [x for x in self.db if x == identifier][0] # Get factor from set
    
    def __len__(self) -> int:
        return len(self.db)
    
    

class Optimizer:
    def __init__(self, relin_skip = None, relin_thres = None) -> None:
        self.factor_db = FactorDatabase()
        self.new_nodes = Values()

        self.smart_params = SmartProjectionParams()
        self.smart_params.setDegeneracyMode(DegeneracyMode.ZERO_ON_DEGENERACY)
        self.smart_params.setLinearizationMode(LinearizationMode.HESSIAN)

        self.smart_pixel_noise = noiseModel.Isotropic.Sigma(2, 1.5) # Pixel std deviation in u and v
        self.generic_pixel_noise = noiseModel.Robust.Create(
            noiseModel.mEstimator.Huber.Create(1.345),
            self.smart_pixel_noise
        )

        self.isam_parameters = ISAM2Params()
        if relin_skip is not None:
            self.isam_parameters.relinearizeSkip = relin_skip
        if relin_thres is not None:
            self.isam_parameters.setRelinearizeThreshold(relin_thres)
        self.isam = ISAM2(self.isam_parameters)

    def _add_factor(self, factor, identifier: str) -> None:
        self.factor_db.add_pending(identifier, factor) # Add factor as pending to database

    def add_node(self, value: 'Geometry', id: int, node_type: NodeType) -> None:
        self.new_nodes.insert(node_type(id), value)

    def add_between_factor(self, pose_from: int, pose_to: int, node_type: NodeType, measurement: 'Geometry.SE3', sigmas: list) -> None:
        noise_model = noiseModel.Diagonal.Sigmas(sigmas)
        from_key, to_key = node_type(pose_from), node_type(pose_to)
        self._add_factor(BetweenFactorPose3(from_key, to_key, measurement, noise_model), f'Between{from_key}-{to_key}')

    def add_velocity_extrinsic_factor(self, pose_from: int, pose_to: int, camera_id: int, node_type: NodeType, measurement: 'Geometry.State', dt: float, sigmas: list) -> None:
        noise_model = noiseModel.Diagonal.Sigmas(sigmas)
        from_key, to_key = node_type(pose_from), node_type(pose_to)
        extrinsics_key = NodeType.EXTRINSIC(camera_id)
        self._add_factor(VelocityExtrinsicFactor(from_key, to_key, extrinsics_key, measurement, dt, noise_model), f'VelocityExtrinsic{from_key}-{extrinsics_key}-{to_key}')

    def add_velocity_factor(self, pose_from: int, pose_to: int, node_type: NodeType, measurement: 'Geometry.Vector6', sigmas: list) -> None:
        noise_model = noiseModel.Diagonal.Sigmas(sigmas)
        from_key, to_key = node_type(pose_from), node_type(pose_to)
        self._add_factor(VelocityFactor(from_key, to_key, measurement, noise_model), f'Velocity{from_key}-{to_key}')

    def add_pose_equality(self, value: 'Geometry.SE3', pose_id: int, node_type: NodeType) -> None:
        key = node_type(pose_id)
        self._add_factor(NonlinearEqualityPose3(key, value), f'Equality{key}')

    def add_prior(self, value, id: int, node_type: NodeType, sigma: list) -> None:
        noise_model = noiseModel.Diagonal.Sigmas(sigma)
        key = node_type(id)
        factor = PriorFactorMap(node_type)
        self._add_factor(factor(key, value, noise_model), f'Prior{key}')

    def add_camera_between_factor(self, ref_index: int, camera_id: int, camera_pose_id: int, sigmas: list) -> None:
        noise_model = noiseModel.Diagonal.Sigmas(sigmas)
        self._add_factor(BetweenFactorCamera(X(ref_index), T(camera_id), C(camera_pose_id), noise_model), f'RefExtCam{ref_index}-{camera_id}-{camera_pose_id}')

    def add_reference_anchor(self, camera_id: int, camera_pose_id: int, ref_pose: 'Geometry.SE3', sigmas: list) -> None:
        noise_model = noiseModel.Diagonal.Sigmas(sigmas)
        self._add_factor(ReferenceAnchor(T(camera_id), C(camera_pose_id), ref_pose, noise_model), f'RefAnchor{camera_id}-{camera_pose_id}')

    def add_hand_eye_factor(self, pose_from: int, pose_to: int, camera_id: int, state_from: Geometry.State, state_to: Geometry.State, node_type: NodeType, sigmas: list) -> None:
        noise_model = noiseModel.Diagonal.Sigmas(sigmas)
        from_key, to_key = node_type(pose_from), node_type(pose_to)
        extr_key = NodeType.EXTRINSIC(camera_id)
        self._add_factor(HandEyeFactor(from_key, to_key, extr_key, state_from, state_to, noise_model), f'HandEye{from_key}-{to_key}-{extr_key}')


    def add_extrinsic_projection_factor(self, map_point_id: int, pixels: tuple, pose_id: int, camera: 'Camera', delay=2) -> None:
        landmark = f'MapPoint{map_point_id}'
        new_observation = self.factor_db.add_map_point_observation(landmark, pixels, pose_id)
        if not new_observation: return

        observations = self.factor_db.observations(landmark)
        if len(observations) < delay: return # Need x or more observations before adding to FG

        identifier = f'{landmark}-{NodeType.REFERENCE(pose_id)}'

        if len(observations) > delay:
            factor = ExtrinsicProjectionFactor(NodeType.REFERENCE(pose_id), NodeType.EXTRINSIC(camera.id), NodeType.LANDMARK(map_point_id),
                                               camera, pixels, self.generic_pixel_noise)
            self._add_factor(factor, identifier)
        else:
            for (pixels, pose_id) in observations:
                identifier = f'{landmark}-{NodeType.REFERENCE(pose_id)}'
                factor = ExtrinsicProjectionFactor(NodeType.REFERENCE(pose_id), NodeType.EXTRINSIC(camera.id), NodeType.LANDMARK(map_point_id),
                                               camera, pixels, self.generic_pixel_noise)
                self._add_factor(factor, identifier)


    def add_projection_factor(self, map_point_id: int, pixels: tuple, pose_id: int, node_type: NodeType, camera: 'Camera', Trc = None) -> None:
        landmark = f'MapPoint{map_point_id}'
        new_observation = self.factor_db.add_map_point_observation(landmark, pixels, pose_id)
        if not new_observation: return

        observations = self.factor_db.observations(landmark)
        x = 2
        if len(observations) < x: return # Need x or more observations before adding to FG
        if Trc is None: Trc = Geometry.SE3.NED_to_RDF_map()

        K = Cal3_S2(*camera.parameters_with_skew)
        identifier = f'{landmark}-{node_type(pose_id)}'
        if len(observations) > x:
            factor = GenericProjectionFactorCal3_S2(pixels, self.generic_pixel_noise, node_type(pose_id), L(map_point_id), K, Trc)
            self._add_factor(factor, identifier)
        else:
            for (pixels, pose_id) in observations:
                identifier = f'{landmark}-{node_type(pose_id)}'
                factor = GenericProjectionFactorCal3_S2(pixels, self.generic_pixel_noise, node_type(pose_id), L(map_point_id), K, Trc)
                self._add_factor(factor, identifier)


    def add_smart_projection_factor(self, map_point_id: int, pixels: tuple, pose_id: int, camera: 'Camera', Trc = None) -> None:
        identifier = f'MapPoint{map_point_id}'

        # if len(self.factor_db.observations(identifier)) > 50: return
        is_new_observation = self.factor_db.add_map_point_observation(identifier, pixels, pose_id) # Include pixel observation in factor DB
        if not is_new_observation: return

        observations = self.factor_db.observations(identifier) # Get observations from factor DB
        if Trc is None: Trc = Geometry.SE3.NED_to_RDF_map()

        K = Cal3_S2(*camera.parameters_with_skew)
        factor = SmartProjectionPoseFactorCal3_S2(self.smart_pixel_noise, K, Trc, self.smart_params)
        # factor = SmartProjectionPose3Factor(self.smart_pixel_noise, K, Trc, self.smart_params)
        for (kp, pose_id) in observations: factor.add(kp, C(pose_id)) # Add all observations to the factor
        
        # If factor already exists, remove/mark for removal
        if identifier in self.factor_db: self.factor_db.remove(identifier)

        # Add factor to factor graph
        self._add_factor(factor, identifier)

    def optimize(self, extra_updates: int = 0) -> None:
        new_factors, removal_indeces = self.factor_db.prepare_update()
        
        # Perform incremental update to iSAM, and remove/readd nodes to be updated
        try:
            self.isam.update(new_factors, self.new_nodes, removal_indeces)
            for _ in range(extra_updates): self.isam.update() # Perform extra updates
        except Exception as e:
            print(f'iSAM Update failed!!! {e}')

        self.new_nodes.clear() # Clear list of new nodes, since they are now added
        self.factor_db.finish_update() # Mark factor update as complete

    def get_node_estimate(self, id: int, node_type: NodeType) -> 'Geometry':
        key = node_type(id)
        values = None
        if self.new_nodes.exists(key): values = self.new_nodes
        if self.current_estimate.exists(key): values = self.current_estimate
        if values is None: return values

        match node_type:
            case NodeType.CAMERA: estimate = values.atPose3(key)
            case NodeType.REFERENCE: estimate = values.atPose3(key)
            case NodeType.EXTRINSIC: estimate = values.atPose3(key)
            case NodeType.LANDMARK: estimate = values.atPoint3(key)
        return estimate

    def get_visualization_variables(self) -> tuple[NonlinearFactorGraph, Values]:
        return self.factor_db.master_graph, self.current_estimate

    @property
    def current_estimate(self) -> Values:
        return self.isam.calculateBestEstimate()
