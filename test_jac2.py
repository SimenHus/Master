import symforce.symbolic as sf

T_rel = sf.Pose3(
    R = sf.Rot3.symbolic('R_rel'),
    t = sf.V3.symbolic('t_rel')
)

T_ref = sf.Pose3(
    R = sf.Rot3.symbolic('R_ref'),
    t = sf.V3.symbolic('t_ref')
)


landmark = sf.V3.symbolic('L')

T_cam = T_ref.compose(T_rel)


pred = T_cam.inverse() * landmark
jac = pred.jacobian(T_rel)

subs = {
    'R_rel_w': 0,
    'R_rel_x': 0,
    'R_rel_y': 0,
    'R_rel_z': 1,
    't_rel0': 0,
    't_rel1': 3,
    't_rel2': 0,

    'R_ref_w': 1,
    'R_ref_x': 0,
    'R_ref_y': 0,
    'R_ref_z': 0,
    't_ref0': 1,
    't_ref1': 0,
    't_ref2': 0,

    'L0': 5,
    'L1': 0,
    'L2': 0
}

subbed = jac.subs(subs)
print(subbed)