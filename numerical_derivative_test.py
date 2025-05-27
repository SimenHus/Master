

# Custom factor documentation: https://github.com/borglab/gtsam/blob/develop/gtsam/nonlinear/doc/CustomFactor.ipynb



import numpy as np
from gtsam import CustomFactor, noiseModel, Values, Pose2, Pose3, Rot3
from gtsam.utils.numerical_derivative import numericalDerivative31, numericalDerivative32, numericalDerivative33
from gtsam.utils.numerical_derivative import numericalDerivative21, numericalDerivative22

from src.backend.factors import BetweenFactorCamera, ReferenceAnchor, KinematicCameraFactor
from src.util import Geometry

noise_model = noiseModel.Isotropic.Sigma(6, 0.1)
# custom_factor = BetweenFactorCamera(0, 1, 2, noise_model)
# custom_factor = ReferenceAnchor(0, 1, T_rel, noise_model)
state = Geometry.State(
    velocity=np.array([1, 1, 0]),
    acceleration=np.array([0, 0, 0]),
    attrate=np.array([0.1, 0.1, 0]) * np.pi/180,
)
dt = 0.5
custom_factor = KinematicCameraFactor(0, 1, 2, state, dt, noise_model)

T_rel = Pose3.Expmap([2.0, 0.0, -1.0, 10, -5, 1])
xi_c = T_rel.inverse().AdjointMap() @ state.twist * dt

Twr1 = Pose3()
Twr2 = Twr1.compose(Pose3.Expmap(state.twist*dt))
Twc1 = Twr1.compose(T_rel)
Twc2 = Twr2.compose(T_rel)

T1, T2 = Twc1, Twc2


values = Values()
values.insert(0, T1)
values.insert(1, T2)
values.insert(2, T_rel)

# Allocate the Jacobians and call error_func
H = [np.empty((6, 6), order='F'),np.empty((6, 6), order='F'), np.empty((6, 6), order='F')]
custom_factor.evaluateError('', values, H)

def f(*args):
    v = Values()
    for i, arg in enumerate(args): v.insert(i, arg)
    return custom_factor.evaluateError('', v)

numerical0 = numericalDerivative31(f, values.atPose3(0), values.atPose3(1), values.atPose3(2))
numerical1 = numericalDerivative32(f, values.atPose3(0), values.atPose3(1), values.atPose3(2))
numerical2 = numericalDerivative33(f, values.atPose3(0), values.atPose3(1), values.atPose3(2))

# numerical0 = numericalDerivative21(f, values.atPose3(0), values.atPose3(1))
# numerical1 = numericalDerivative22(f, values.atPose3(0), values.atPose3(1))

# Check the numerical derivatives against the analytical ones
np.testing.assert_allclose(H[0], numerical0, rtol=1e-5, atol=1e-8)
np.testing.assert_allclose(H[1], numerical1, rtol=1e-5, atol=1e-8)
np.testing.assert_allclose(H[2], numerical2, rtol=1e-5, atol=1e-8)