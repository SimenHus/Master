
from gtsam import Pose3, Rot3, Point3
from gtsam import Pose2, Rot2, Point2

class SE2:
    pose = Pose2
    rot = Rot2
    point = Point2
    dim_t = 2
    dim_r = 1

class SE3:
    pose = Pose3
    rot = Rot3
    point = Point3
    dim_t = 3
    dim_r = 3


class GroupIdentifier:

    @staticmethod
    def identify(state: Pose3 | Pose2) -> SE3 | SE2:
        if type(state) == Pose3: return SE3
        if type(state) == Pose2: return SE2