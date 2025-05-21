

# Custom factor documentation: https://github.com/borglab/gtsam/blob/develop/gtsam/nonlinear/doc/CustomFactor.ipynb



import numpy as np
from gtsam import CustomFactor, noiseModel, Values, Pose2, Pose3
from gtsam.utils.numerical_derivative import numericalDerivative31, numericalDerivative32, numericalDerivative33

from src.backend.factors import BetweenFactorCamera

noise_model = noiseModel.Isotropic.Sigma(6, 0.1)
custom_factor = BetweenFactorCamera(0, 1, 2, noise_model)

T_ref = Pose3()
T_rel = Pose3.Expmap([2.0, 0.0, -1.0, 10, -5, 1])
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