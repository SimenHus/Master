

from src.simulation import TrajectoryGenerator
from src.visualization import LiveTrajectory3D
from src.backend import NodeType
from src.util import Geometry

from gtsam import Values
import numpy as np

def generate_mps() -> list[Geometry.Point3]:
    mps = [
        np.array([10, 10, 10]),
        np.array([-10, 10, 10]),
        np.array([10, -10, 10]),
        np.array([-10, -10, 10]),
        np.array([10, 10, -10]),
        np.array([-10, 10, -10]),
        np.array([10, -10, -10]),
        np.array([-10, -10, -10]),
        np.array([5, 5, 5]),
        np.array([-5, 5, 5]),
        np.array([5, -5, 5]),
        np.array([-5, -5, 5]),
        np.array([5, 5, -5]),
        np.array([-5, 5, -5]),
        np.array([5, -5, -5]),
        np.array([-5, -5, -5]),
    ]
    return mps


mps = generate_mps()
traj = TrajectoryGenerator.circular(w=1)
traj2 = TrajectoryGenerator.circular(w=0)
plot_3D = LiveTrajectory3D(NodeType.REFERENCE, delay=0.1)

result = Values()

for i, mp in enumerate(mps):
    result.insert(NodeType.LANDMARK(i), mp)


for i, pose in enumerate(traj):
    result.insert(NodeType.REFERENCE(i), pose)
    plot_3D.update(result)

plot_3D.finished()