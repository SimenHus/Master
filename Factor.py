
import gtsam.noiseModel
from common import *



class DistanceFactor2D(gtsam.CustomFactor):
    def __init__(self, key, noise_model, measured_distance, known_point):
        # Call parent constructor
        super().__init__(noise_model, [key], self.error)

        self.measured_distance = measured_distance
        self.known_point = np.array(known_point)
    
    def error(self, _, values: gtsam.Values, H: gtsam.JacobianVector | None = None):
        # Get the current value of the variable
        x, y = values.atVector(self.keys()[0])
        current_point = np.array([x, y])
        
        # Compute the residual
        predicted_distance = np.linalg.norm(current_point - self.known_point)
        residual = predicted_distance - self.measured_distance

        if H is not None:
            H[0] = np.eye(1)

        return np.array([residual])
    


class CameraExtrinsicFactor(gtsam.CustomFactor):
    def __init__(self, key_base_pose, key_extrinsic_transform, noise_model, observed_point, expected_point) -> None:
        super().__init__(noise_model, [key_base_pose, key_extrinsic_transform], self.error)
        self.observed_point = observed_point  # Point in the camera frame
        self.expected_point = expected_point  # Expected point in the base frame

    def error(self, _, values: gtsam.Values, H: gtsam.JacobianVector | None):
        base_pose = values.atPose3(self.keys()[0])
        extrinsic_transform = values.atPose3(self.keys()[1])
        # Transform the observed point from the camera frame to the base frame
        transformed_point = base_pose.compose(extrinsic_transform).transformFrom(self.observed_point)

        # Calculate the error as the difference between the transformed point and the measured point
        error = transformed_point - self.expected_point

        # If Jacobians are requested, set them to identity for simplicity
        if H is not None:
            H[0] = np.eye(3)
            H[1] = np.eye(3)

        return error