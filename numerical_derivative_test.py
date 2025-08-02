

# Custom factor documentation: https://github.com/borglab/gtsam/blob/develop/gtsam/nonlinear/doc/CustomFactor.ipynb



import numpy as np
from gtsam import CustomFactor, noiseModel, Values, Pose2, Pose3, Rot3
from gtsam.utils.numerical_derivative import numericalDerivative31, numericalDerivative32, numericalDerivative33
from gtsam.utils.numerical_derivative import numericalDerivative21, numericalDerivative22

from src.backend.factors import BetweenFactorCamera, ReferenceAnchor, VelocityExtrinsicFactor, VelocityFactor, HandEyeFactor, ExtrinsicProjectionFactor
from src.util import Geometry
from src.structs import Camera

noise_model = noiseModel.Isotropic.Sigma(6, 0.1)
# T_rel = Pose3.Expmap([2.0, 0.0, -1.0, 10, -5, 1])
T_rel = Geometry.SE3.from_vector(np.array([32, -26, 89.1, 3, -2, -5]), radians=False)
state = Geometry.State(
    velocity=np.array([1, 1, 0]),
    acceleration=np.array([0, 0, 0]),
    attrate=np.array([0.1, 0.1, 0]) * np.pi/180,
)
noise = Pose3.Expmap([0., 0.1, -0.1, 0.1, -0.1, 0.0])

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

    return [np.empty((6, 6), order='F'), np.empty((6, 6), order='F')]

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
    custom_factor = VelocityExtrinsicFactor(0, 1, 2, state, dt, noise_model)

    return [np.empty((6, 6), order='F'), np.empty((6, 6), order='F'), np.empty((6, 6), order='F')]

def velocity():
    global custom_factor, noise
    dt = 0.5
    xi = state.twist * dt
    Twx1 = Pose3()
    Twx2 = Twx1.compose(Pose3.Expmap(xi)).compose(noise)
    values.insert(0, Twx1)
    values.insert(1, Twx2)
    custom_factor = VelocityFactor(0, 1, xi, noise_model)
    return [np.empty((6, 6), order='F'), np.empty((6, 6), order='F')]


def projection():
    global custom_factor, noise
    smart_pixel_noise = noiseModel.Isotropic.Sigma(2, 1.5)
    generic_pixel_noise = noiseModel.Robust.Create(
        noiseModel.mEstimator.Huber.Create(1.345),
        smart_pixel_noise
    )
    Twr = Geometry.SE3.from_vector(np.array([10, -3, 62.4, 0.2, 1, -2]), radians=False)
    Trc = T_rel
    Twc = Twr.compose(Trc)
    landmark = np.array([20, 10, 1])
    landmark_c = Twc.transformTo(landmark)
    values.insert(0, Twr)
    values.insert(1, Trc)
    values.insert(2, landmark)
    camera = Camera([100., 100., 50., 50.], [])
    pixels = camera.project(landmark_c)
    custom_factor = ExtrinsicProjectionFactor(0, 1, 2, camera, pixels, generic_pixel_noise)
    return [np.empty((6, 6), order='F'), np.empty((6, 6), order='F'), np.empty((2, 3), order='F')]


def ha_calib():
    global custom_factor, noise

    Trc = T_rel

    # Tr1r2
    odom_r = Pose3(Rot3.RzRyRx(np.array([0, 0, 30])*np.pi/180), np.array([-3, 0, 0]))
    odom_c = Trc.inverse().compose(odom_r).compose(Trc)

    Tr1 = Pose3()
    Tc1 = Tr1.compose(Trc)
    Tr2 = Tr1.compose(odom_r)
    Tc2 = Tr2.compose(Trc).compose(noise)

    v1 = Geometry.SE3.as_vector(Tr1, show_degrees=False)
    v2 = Geometry.SE3.as_vector(Tr2, show_degrees=False)
    s1 = Geometry.State(position=v1[3:], attitude=v1[:3])
    s2 = Geometry.State(position=v2[3:], attitude=v2[:3])


    values.insert(0, Tc1)
    values.insert(1, Tc2)
    values.insert(2, T_rel)

    custom_factor = HandEyeFactor(0, 1, 2, s1, s2, noise_model)
    return [np.empty((6, 6), order='F'), np.empty((6, 6), order='F'), np.empty((6, 6), order='F')]

H = projection()
custom_factor.evaluateError('', values, H)

def f(*args):
    v = Values()
    for i, arg in enumerate(args): v.insert(i, arg)
    return custom_factor.evaluateError('', v)

# OBS OBS OBS
if True:
    numerical0 = numericalDerivative31(f, values.atPose3(0), values.atPose3(1), values.atPoint3(2))
    numerical1 = numericalDerivative32(f, values.atPose3(0), values.atPose3(1), values.atPoint3(2))
    numerical2 = numericalDerivative33(f, values.atPose3(0), values.atPose3(1), values.atPoint3(2))
    nums = [numerical0, numerical1, numerical2]

elif len(H) == 3:
    numerical0 = numericalDerivative31(f, values.atPose3(0), values.atPose3(1), values.atPose3(2))
    numerical1 = numericalDerivative32(f, values.atPose3(0), values.atPose3(1), values.atPose3(2))
    numerical2 = numericalDerivative33(f, values.atPose3(0), values.atPose3(1), values.atPose3(2))

    nums = [numerical0, numerical1, numerical2]
elif len(H) == 2:
    numerical0 = numericalDerivative21(f, values.atPose3(0), values.atPose3(1))
    numerical1 = numericalDerivative22(f, values.atPose3(0), values.atPose3(1))

    nums = [numerical0, numerical1]



# Check the numerical derivatives against the analytical ones
for i in range(len(H)):
    print(f'Testing: {i}')
    np.testing.assert_allclose(H[i], nums[i], rtol=1e-5, atol=1e-8)
    print(f'Test {i} was a success')
