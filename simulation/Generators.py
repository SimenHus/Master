
from gtsam import Pose3, Rot3, Point3
from gtsam import Pose2, Rot2, Point2

class Generator2D:
    pose = Pose2
    rot = Rot2
    point = Point2

class Generator3D:
    pose = Pose3
    rot = Rot3
    point = Point3