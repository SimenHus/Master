
from gtsam import CustomFactor, noiseModel, Pose3, Pose2, PriorFactorPoint3, Point3
from gtsam import traits
import numpy as np

class PositionFactor2D(CustomFactor):
    def __init__(self, key: int, measurement: Pose2, noise_model: noiseModel = None):
        super().__init__(noise_model, [key], self.evaluateError)
        self.measurement = measurement.translation()

    def evaluateError(self, _, values, H = None):
        T = values.atPose2(self.keys()[0])
        R = T.rotation()
        if H is not None:
            # H[0] = np.array([
            #     [R.c(), -R.s(), 0.0],
            #     [R.s(), R.c(), 0.0]
            # ])
            H[0] = np.array([
                [1, 0, 0],
                [0, 1, 0]
            ])
        return T.translation() - self.measurement



class CameraExtrinsicFactor3D(CustomFactor):
    def __init__(self, from_key: int, to_key: int, measurement: Pose3, noise_model: noiseModel = None) -> None:
        super().__init__(noiseModel, [from_key, to_key], self.evaluateError)
        self.measurement = measurement


#    Vector evaluateError(const T& p1, const T& p2, boost::optional<Matrix&> H1 =
#        boost::none, boost::optional<Matrix&> H2 = boost::none) const {
#        T hx = traits<T>::Between(p1, p2, H1, H2); // h(x)
#        // manifold equivalent of h(x)-z -> log(z,h(x))
#  #ifdef SLOW_BUT_CORRECT_BETWEENFACTOR
#        typename traits<T>::ChartJacobian::Jacobian Hlocal;
#        Vector rval = traits<T>::Local(measured_, hx, boost::none, (H1 || H2) ? &Hlocal : 0);
#        if (H1) *H1 = Hlocal * (*H1);
#        if (H2) *H2 = Hlocal * (*H2);
#        return rval;
    def evaluateError(self, _, values, H = None):
        T_world_ref = self.measurement # Measurement of reference frame location in world frame
        T_ref_camera = values.atPose3(self.keys()[0]) # Current estimate of camera extrinsics from reference frame
        T_world_camera = values.atPose3(self.keys()[1]) # Current estimate of camera in world frame

        hx = traits.between(T_world_ref, T_world_camera)
        rval = traits.Local(T_ref_camera, hx)
        if H is not None:
            Hlocal = traits.ChartJacobian.Jacobian
            H[0] = Hlocal * H[0]
            H[1] = Hlocal * H[1]
        return rval


class PositionFactor3D(CustomFactor):
    def __init__(self, key: int, measurement: Pose3, noise_model: noiseModel = None):
        super().__init__(noise_model, [key], self.evaluateError)
        self.measurement = measurement.translation()

    def evaluateError(self, _, values, H = None):
        T = values.atPose3(self.keys()[0])
        R = T.rotation().matrix()
        if H is not None:
            H[0] = np.array([
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ])
        return T.translation() - self.measurement
    

