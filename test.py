


import numpy as np
from src.util import Geometry

roll, pitch, yaw = np.array([10, 13, -4]) * np.pi / 180
R1 = Geometry.SO3.from_vector([roll, pitch, yaw], radians=True)
R2 = Geometry.SO3.Ypr(yaw, pitch, roll)
R3 = Geometry.SO3.from_vector([5, 0, 0], radians=False)




print(Geometry.SO3.as_vector(R1, radians=False))

print(Geometry.SO3.as_vector(R1.compose(R3), radians=False))