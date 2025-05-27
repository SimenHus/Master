

# Custom factor documentation: https://github.com/borglab/gtsam/blob/develop/gtsam/nonlinear/doc/CustomFactor.ipynb



import numpy as np
from gtsam import CustomFactor, noiseModel, Values, Pose2, Pose3, Rot3
from gtsam.utils.numerical_derivative import numericalDerivative31, numericalDerivative32, numericalDerivative33
from gtsam.utils.numerical_derivative import numericalDerivative21, numericalDerivative22

from src.backend.factors import BetweenFactorCamera, ReferenceAnchor, KinematicCameraFactor
from src.util import Geometry

noise_model = noiseModel.Isotropic.Sigma(6, 0.1)
T_rel = Pose3.Expmap([2.0, 0.0, -1.0, 10, -5, 1])
state = Geometry.State(
    velocity=np.array([1, 1, 0]),
    acceleration=np.array([0, 0, 0]),
    attrate=np.array([0.1, 0.1, 0]) * np.pi/180,
)


values = Values()
custom_factor = None

def between():
    global custom_factor
    T1 = Pose3()
    T2 = T1.compose(T_rel)
    values.insert(0, T1)
    values.insert(1, T_rel)
    values.insert(2, T2)
    custom_factor = BetweenFactorCamera(0, 1, 2, noise_model)

    return [np.empty((6, 6), order='F'),np.empty((6, 6), order='F'), np.empty((6, 6), order='F')]



def anchor():
    global custom_factor
    T1 = Pose3()
    custom_factor = ReferenceAnchor(0, 1, T1, noise_model)
    values.insert(0, T_rel)
    values.insert(1, T1.compose(T_rel))

    return [np.empty((6, 6), order='F'),np.empty((6, 6), order='F')]

def kinematic():
    global custom_factor
    dt = 0.5
    xi_c = T_rel.inverse().AdjointMap() @ state.twist * dt

    Twr1 = Pose3()
    Twr2 = Twr1.compose(Pose3.Expmap(state.twist*dt))
    Twc1 = Twr1.compose(T_rel)
    Twc2 = Twr2.compose(T_rel)

    values.insert(0, Twc1)
    values.insert(1, Twc2)
    values.insert(2, T_rel)
    custom_factor = KinematicCameraFactor(0, 1, 2, state, dt, noise_model)

    return [np.empty((6, 6), order='F'), np.empty((6, 6), order='F'), np.empty((6, 6), order='F')]

H = kinematic()
custom_factor.evaluateError('', values, H)

def f(*args):
    v = Values()
    for i, arg in enumerate(args): v.insert(i, arg)
    return custom_factor.evaluateError('', v)

if len(H) == 3:
    numerical0 = numericalDerivative31(f, values.atPose3(0), values.atPose3(1), values.atPose3(2))
    numerical1 = numericalDerivative32(f, values.atPose3(0), values.atPose3(1), values.atPose3(2))
    numerical2 = numericalDerivative33(f, values.atPose3(0), values.atPose3(1), values.atPose3(2))

    nums = [numerical0, numerical1, numerical2]


if len(H) == 2:
    numerical0 = numericalDerivative21(f, values.atPose3(0), values.atPose3(1))
    numerical1 = numericalDerivative22(f, values.atPose3(0), values.atPose3(1))

    nums = [numerical0, numerical1]

# Check the numerical derivatives against the analytical ones
for i in range(len(H)):
    print(i)
    np.testing.assert_allclose(H[i], nums[i], rtol=1e-5, atol=1e-8)
