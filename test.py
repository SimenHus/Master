


from src.util import Geometry
import numpy as np


mp = np.array([1,2,3])


pose = Geometry.SE3.from_vector(np.array([0., 0., 90., 5., 5., 3.]), radians=False)

print(pose.transformTo(mp))


print(pose.matrix()@np.append(mp, 1))
print(pose.inverse().matrix()@np.append(mp, 1))