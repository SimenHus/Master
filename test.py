



from gtsam import Pose3, Rot3, Point3

R = Rot3(0, 0, 0, 1)
t = Point3(1, 2, 3)
pose1 = Pose3()
pose2 = Pose3(R, t)


class SE3(Pose3):

    def set_rot(self, rot: Rot3) -> None:
        self.

pose1.