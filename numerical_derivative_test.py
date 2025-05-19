

# Custom factor documentation: https://github.com/borglab/gtsam/blob/develop/gtsam/nonlinear/doc/CustomFactor.ipynb



import numpy as np
from gtsam import CustomFactor, noiseModel, Values, Pose2, Pose3
from gtsam.utils.numerical_derivative import numericalDerivative31, numericalDerivative32, numericalDerivative33

from src.backend.factors import BetweenFactorCamera


# def error_func(this: CustomFactor, v: Values, H: list[np.ndarray]=None):
#     """
#     Error function that mimics a BetweenFactor
#     :param this: reference to the current CustomFactor being evaluated
#     :param v: Values object
#     :param H: list of references to the Jacobian arrays
#     :return: the non-linear error
#     """
#     key0 = this.keys()[0]
#     key1 = this.keys()[1]
#     gT1, gT2 = v.atPose2(key0), v.atPose2(key1)
#     error = measurement.localCoordinates(gT1.between(gT2))

#     if H is not None:
#         result = gT1.between(gT2)
#         H[0] = -result.inverse().AdjointMap()
#         H[1] = np.eye(3)
#     return error


noise_model = noiseModel.Isotropic.Sigma(6, 0.1)
custom_factor = BetweenFactorCamera(0, 1, 2, noise_model)

T_ref = Pose3()
T_rel = Pose3.Expmap([0.0, 0.0, 1.0, 0.5, 0.2, -0.5])
T_cam = T_ref.compose(T_rel)

values = Values()
values.insert(0, T_ref)
values.insert(1, T_rel)
values.insert(2, T_cam)

# Allocate the Jacobians and call error_func
H = [np.empty((6, 6), order='F'),np.empty((6, 6), order='F'), np.empty((6, 6), order='F')]
custom_factor.evaluateError('', values, H)

# We use error_func directly, so we need to create a binary function constructing the values.
def f (T1, T2, T3):
    v = Values()
    v.insert(0, T1)
    v.insert(1, T2)
    v.insert(2, T3)
    return custom_factor.evaluateError('', v)

numerical0 = numericalDerivative31(f, values.atPose3(0), values.atPose3(1), values.atPose3(2))
numerical1 = numericalDerivative32(f, values.atPose3(0), values.atPose3(1), values.atPose3(2))
numerical2 = numericalDerivative33(f, values.atPose3(0), values.atPose3(1), values.atPose3(2))

# Check the numerical derivatives against the analytical ones
np.testing.assert_allclose(H[0], numerical0, rtol=1e-5, atol=1e-8)
np.testing.assert_allclose(H[1], numerical1, rtol=1e-5, atol=1e-8)
np.testing.assert_allclose(H[2], numerical2, rtol=1e-5, atol=1e-8)