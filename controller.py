import numpy as np
import mujoco

# Integration timestep in seconds. This corresponds to the amount of time the joint
# velocities will be integrated for to obtain the desired joint positions.
integration_dt: float = 0.1

# Damping term for the pseudoinverse. This is used to prevent joint velocities from
# becoming too large when the Jacobian is close to singular.
damping: float = 1e-4

# Gains for the twist computation. These should be between 0 and 1. 0 means no
# movement, 1 means move the end-effector to the target in one integration step.
Kpos: float = 0.95
Kori: float = 0.95

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.002

# Nullspace P gain.
Kn = np.asarray([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0])

# Maximum allowable joint velocity in rad/s.
max_angvel = 0.785

def diffik_nullspace(model, data, dx, error_quat = np.zeros(4)):
    #ignore error_quat for now

    joint_names = ["joint1","joint2","joint3","joint4","joint5","joint6", "joint7"]
    dof_ids = np.array([model.joint(name).id for name in joint_names])

    q0 = model.key("home").qpos[dof_ids]

    site_name = "attachment_site"
    site_id = model.site(site_name).id

    # Pre-allocate numpy arrays.
    jac_all = np.zeros((6, model.nv))
    diag = damping * np.eye(6)
    eye = np.eye(len(dof_ids))
    twist = np.zeros(6)

    twist[:3] = Kpos * dx / integration_dt
    mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
    twist[3:] *= Kori / integration_dt

    # Jacobian.
    mujoco.mj_jacSite(model, data, jac_all[:3], jac_all[3:], site_id)

    jac = jac_all[:,:len(dof_ids)]

    # Damped least squares.
    dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, twist)

    # Nullspace control biasing joint velocities towards the home configuration.
    dq += (eye - np.linalg.pinv(jac) @ jac) @ (Kn * (q0 - data.qpos[dof_ids]))

    # Clamp maximum joint velocity.
    dq_abs_max = np.abs(dq).max()
    if dq_abs_max > max_angvel:
        dq *= max_angvel / dq_abs_max

    # Integrate joint velocities to obtain joint positions.
    q = data.qpos.copy()  # Note the copy here is important.
    dq = np.pad(dq, (0, model.nv - len(dof_ids)), mode='constant')
    mujoco.mj_integratePos(model, q, dq, integration_dt)

    np.clip(q[dof_ids], *model.jnt_range.T[:,:len(dof_ids)], out=q[dof_ids])

    return q[:8] 