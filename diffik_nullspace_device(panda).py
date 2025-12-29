import mujoco
import mujoco.viewer
import numpy as np
import time
import pygame

from pynput import keyboard
from PIL import Image

from scipy.spatial.transform import Rotation as R

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

def main() -> None:
    def on_press(key):
        try:
            if key.char == 'p':
                commands["take_picture"] = True
        except AttributeError:
            print("Invalid keyboard press")

    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    #starts new thread so it doesn't block
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Initialize pygame and joystick module
    pygame.init()
    pygame.joystick.init()

    # Detect controllers
    if pygame.joystick.get_count() == 0:
        print("⚠️ No controller detected. Plug in your PS4/PS5 controller and try again.")
        exit()

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"🎮 Detected controller: {joystick.get_name()}")

    robot_path = "C:/Users/gdev/Documents/CS/DL/projects/Robotics/custom_gym_env/robots/franka_emika_panda/pick_place_custom.xml"
    #robot_path = "C:/Users/gdev/Documents/CS/DL/projects/Robotics/mujoco-3.3.3-windows-x86_64/model/SO101/pick_place_custom.xml"

    # Load the model and data.
    model = mujoco.MjModel.from_xml_path(robot_path)
    data = mujoco.MjData(model)

    # Enable gravity compensation. Set to 0.0 to disable.
    model.body_gravcomp[:] = float(gravity_compensation)
    model.opt.timestep = dt

    # End-effector site we wish to control.
    #site_name = "gripperframe"
    site_name = "attachment_site"
    site_id = model.site(site_name).id

    # Get the dof and actuator ids for the joints we wish to control. These are copied
    # from the XML file. Feel free to comment out some joints to see the effect on
    # the controller.
    joint_names = ["joint1","joint2","joint3","joint4","joint5","joint6", "joint7"]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(f'actuator{i}').id for i in range(1,len(joint_names)+1)])

    print(dof_ids)
    print(actuator_ids)

    gripper_joint_name = "finger_joint1" #"actuator8"
    gripper_joint_id = model.joint(gripper_joint_name).id
    gripper_qpos_addr = model.jnt_qposadr[gripper_joint_id]

    wroll_joint_id = model.joint("joint7").id
    wroll_qpos_addr = model.jnt_qposadr[wroll_joint_id]
    wroll_min = model.jnt_range[wroll_joint_id][0]
    wroll_max = model.jnt_range[wroll_joint_id][1]

    # Initial joint configuration saved as a keyframe in the XML file.
    key_name = "home"
    key_id = model.key(key_name).id
    q0 = model.key(key_name).qpos[dof_ids]

    # Mocap body we will control with our mouse.
    #mocap_name = "target"
    #mocap_id = model.body(mocap_name).mocapid[0]

    # Pre-allocate numpy arrays.
    jac_all = np.zeros((6, model.nv))
    diag = damping * np.eye(6)
    eye = np.eye(len(dof_ids))
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)

    commands = {
        "take_picture": False,
        "reset": False
    }
    # Create a renderer once (reused on each capture)
    renderer = mujoco.Renderer(model, width=640, height=480)
    # Choose the camera you want to capture from
    cam_id = model.camera("fixed_cam").id    # or 0, 1, etc.

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=True,
        show_right_ui=True,
    ) as viewer:
        # Reset the simulation.
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Reset the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        mujoco.mj_forward(model, data)

        target_pos = data.site(site_id).xpos.copy()
        target_quat = np.zeros(4)
        mujoco.mju_mat2Quat(target_quat, data.site(site_id).xmat)

        # Enable site frame visualization.
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        while viewer.is_running():
            step_start = time.time()

            if commands['take_picture']:
                commands['take_picture'] = False

                # Render from the camera
                renderer.update_scene(data, camera=cam_id)
                rgb = renderer.render()

                Rmat = data.site(site_id).xmat.reshape(3, 3)
                # Convert to Euler angles
                site_euler = R.from_matrix(Rmat).as_euler("xyz", degrees=True)

                # Save the image
                np.savez(
                    f"outputs/capture_{time.time()}.npz",
                    img = rgb,
                    state = np.concat([data.site(site_id).xpos.copy(), site_euler, np.array([data.qpos[7]])])
                )
                #Image.fromarray(rgb).save(f"outputs/capture_{time.time()}.png")

                print("Picture saved")

            pygame.event.pump()

            step_size = .001
            grip_delta = 25
            wroll_delta = 0.1
            # Read joystick axes
            dx = joystick.get_axis(0)   # Left stick X
            dy = -joystick.get_axis(1)  # Left stick Y
            dz = -joystick.get_axis(3)  # Right stick vertical

            if abs(dx) < 0.04:
                dx = 0
            if abs(dy) < 0.04:
                dy = 0
            if abs(dz) < 0.04:
                dz = 0

            delta = np.array([dx, dy, dz]) * step_size
            target_pos += delta

            #print(delta)

            # Spatial velocity (aka twist).
            dx = target_pos - data.site(site_id).xpos
            twist[:3] = Kpos * dx / integration_dt
            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, target_quat, site_quat_conj)
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

            L1_pressed = joystick.get_button(9)  # L1
            R1_pressed = joystick.get_button(10)  # R1

            gripper_opening = data.ctrl[7]
            # Update gripper opening
            if R1_pressed:
                #print("R1 pressed")
                gripper_opening += grip_delta
            elif L1_pressed:
                #print("L1 pressed")
                gripper_opening -= grip_delta
            # Clamp to valid range
            gripper_opening = np.clip(gripper_opening, 0, 255)

            wroll_angle = q[wroll_qpos_addr]
            # Update gripper opening
            if joystick.get_button(13):
                wroll_angle += wroll_delta
            elif joystick.get_button(14):
                wroll_angle -= wroll_delta

            # Clamp to valid range
            wroll_angle = np.clip(wroll_angle, wroll_min, wroll_max)

            data.ctrl[7] = gripper_opening    

            # Set the control signal and step the simulation.
            data.ctrl[actuator_ids] = q[dof_ids]
            mujoco.mj_step(model, data)

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
