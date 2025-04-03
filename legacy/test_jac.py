
import jax.numpy as jnp
import jax
import numpy as np

from gtsam import Rot3, Pose3, Point3


def prediction(T_rel, T_ref, landmark):

    landmark = jnp.append(landmark, jnp.array([1]))
    # T_cam = T_ref.compose(T_rel)
    T_cam = T_ref @ T_rel
    # pred = T_cam.transformTo(landmark)
    pred = (jnp.linalg.inv(T_cam) @ landmark)[:3]
    
    return pred



Tref = jnp.array(Pose3(Rot3(1, 0, 0, 0), Point3(1, 0, 0)).matrix(), dtype=float)
Trel = jnp.array(Pose3(Rot3(1, 0, 0, 0), Point3(1, 0, 0)).matrix(), dtype=float)
p = jnp.array([5, 0, 0], dtype=float)

jac = jax.jacobian(prediction, argnums=[0, 1])(Trel, Tref, p)

print(jac)