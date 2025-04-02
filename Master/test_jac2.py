import symforce.symbolic as sf

# T_rel = sf.Pose3(
#     R = sf.Rot3.symbolic('R_rel'),
#     t = sf.V3.symbolic('t_rel')
# )

# T_ref = sf.Pose3(
#     R = sf.Rot3.symbolic('R_ref'),
#     t = sf.V3.symbolic('t_ref')
# )


T_rel = sf.Pose3.symbolic('T_rel')
T_ref = sf.Pose3.symbolic('T_ref')



landmark = sf.V3.symbolic('L')

T_cam = T_ref.compose(T_rel)


pred = T_cam.inverse() * landmark
jac = pred.jacobian(T_rel)

# Extract symbolic variables dynamically
subs = {}

# Substitute rotation components
subs.update({k: v for k, v in zip(T_rel.R.to_storage(), sf.Pose3().R.to_storage())})
subs.update({k: v for k, v in zip(T_ref.R.to_storage(), sf.Pose3().R.to_storage())})

# Substitute translation components
subs.update({k: v for k, v in zip(T_rel.t, sf.Pose3().t)})
subs.update({k: v for k, v in zip(T_ref.t, sf.Pose3().t)})

# Substitute landmark components
subs.update({k: v for k, v in zip(landmark, sf.V3())})

print(T_rel.R.to_storage())
print(type(jac))
subbed = jac.subs(subs)
print(subbed)