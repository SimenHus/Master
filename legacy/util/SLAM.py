import numpy as np
from gtsam import Values

def determine_loop_closure(odom: np.ndarray, current_estimate: Values,
    key: int, xy_tol=0.6, theta_tol=17) -> int:
    """Simple brute force approach which iterates through previous states
    and checks for loop closure.

    Args:
        odom: Vector representing noisy odometry (x, y, theta) measurement in the body frame.
        current_estimate: The current estimates computed by iSAM2.
        key: Key corresponding to the current state estimate of the robot.
        xy_tol: Optional argument for the x-y measurement tolerance, in meters.
        theta_tol: Optional argument for the theta measurement tolerance, in degrees.
    Returns:
        k: The key of the state which is helping add the loop closure constraint.
            If loop closure is not found, then None is returned.
    """
    if current_estimate:
        t = odom.translation()
        r = odom.rotation().theta()

        prev_est = current_estimate.atPose2(key+1)
        rotated_odom = prev_est.rotation().matrix() @ t
        curr_xy = np.array([prev_est.x() + rotated_odom[0],
                            prev_est.y() + rotated_odom[1]])
        curr_theta = prev_est.theta() + r
        for k in range(1, key+1):
            pose_xy = np.array([current_estimate.atPose2(k).x(),
                                current_estimate.atPose2(k).y()])
            pose_theta = current_estimate.atPose2(k).theta()
            if (abs(pose_xy - curr_xy) <= xy_tol).all() and \
                (abs(pose_theta - curr_theta) <= theta_tol*np.pi/180):
                    return k
