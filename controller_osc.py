import mujoco
import numpy as np

# Cartesian impedance control gains.
impedance_pos = np.asarray([100.0, 100.0, 100.0])  # [N/m]
impedance_ori = np.asarray([50.0, 50.0, 50.0])  # [Nm/rad]

# Joint impedance control gains.
Kp_null = np.asarray([75.0, 75.0, 50.0, 50.0, 40.0, 25.0, 25.0])

# Damping ratio for both Cartesian and joint impedance control.
damping_ratio = 1.0

# Gains for the twist computation. These should be between 0 and 1. 0 means no
# movement, 1 means move the end-effector to the target in one integration step.
Kpos: float = 0.95

# Gain for the orientation component of the twist computation. This should be
# between 0 and 1. 0 means no movement, 1 means move the end-effector to the target
# orientation in one integration step.
Kori: float = 0.95

# Integration timestep in seconds.
integration_dt: float = 1.0

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.002


def osc(model, data, target_pos, q0_key='home'):
    # Compute damping and stiffness matrices.
    damping_pos = damping_ratio * 2 * np.sqrt(impedance_pos)
    damping_ori = damping_ratio * 2 * np.sqrt(impedance_ori)
    Kp = np.concatenate([impedance_pos, impedance_ori], axis=0)
    Kd = np.concatenate([damping_pos, damping_ori], axis=0)
    Kd_null = damping_ratio * 2 * np.sqrt(Kp_null)

    joint_names = ["joint1","joint2","joint3","joint4","joint5","joint6", "joint7"]
    dof_ids = np.array([model.joint(name).id for name in joint_names])

    site_name = "attachment_site"
    site_id = model.site(site_name).id

    q0 = model.key(q0_key).qpos[dof_ids]

    # Pre-allocate numpy arrays.
    jac_full = np.zeros((6, model.nv))
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)
    M_inv_full = np.zeros((model.nv, model.nv))
    Mx = np.zeros((6, 6))

    # Spatial velocity (aka twist).
    dx = target_pos - data.site(site_id).xpos
    twist[:3] = Kpos * dx / integration_dt
    mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
    mujoco.mju_negQuat(site_quat_conj, site_quat)
    error_quat = np.array([1,0,0,0]) #ignore orientation
    #mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
    mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
    twist[3:] *= Kori / integration_dt

    # Jacobian.
    mujoco.mj_jacSite(model, data, jac_full[:3], jac_full[3:], site_id)

    jac = jac_full[:,dof_ids]

    # Compute the task-space inertia matrix.
    mujoco.mj_solveM(model, data, M_inv_full, np.eye(model.nv))

    M_inv = M_inv_full[:len(dof_ids), :len(dof_ids)]

    Mx_inv = jac @ M_inv @ jac.T
    if abs(np.linalg.det(Mx_inv)) >= 1e-2:
        Mx = np.linalg.inv(Mx_inv)
    else:
        Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)

    # Compute generalized forces.
    tau = jac.T @ Mx @ (Kp * twist - Kd * (jac @ data.qvel[dof_ids]))

    # Add joint task in nullspace.
    Jbar = M_inv @ jac.T @ Mx
    ddq = Kp_null * (q0 - data.qpos[dof_ids]) - Kd_null * data.qvel[dof_ids]
    tau += (np.eye(len(dof_ids)) - jac.T @ Jbar.T) @ ddq

    # Add gravity compensation.
    if gravity_compensation:
        tau += data.qfrc_bias[dof_ids]

    # Set the control signal and step the simulation.
    np.clip(tau, *model.actuator_ctrlrange.T[:,:len(dof_ids)], out=tau)

    return tau
