import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

import numpy as np
import mujoco
from mujoco import mj_name2id

from scipy.spatial.transform import Rotation as R

#from controller_diffik import diffik_nullspace
from controller_osc import osc

class PickPlacePandaEnvController(MujocoEnv):
    def __init__(
        self,
        xml_file: str = "scene.xml",
        frame_skip: int = 5, #shouldn't be relevant since we're not going to be controlling the timesteps manually and not via do_simulation
        default_camera_config: dict[str, float | int] = None, #DEFAULT_CAMERA_CONFIG
        initial_pose = 'home',
        control_freq = 20,
        **kwargs
    ):
        observation_space = spaces.Tuple((
            spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),   # ee_pos
            spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),   # ee_euler
            spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)  # normalized gripper state
        ))

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        site_name = "attachment_site"
        self.ee_site_id = self.model.site(site_name).id

        joint_names = ["joint1","joint2","joint3","joint4","joint5","joint6", "joint7"]
        self.dof_ids = np.array([self.model.joint(name).id for name in joint_names])
        self.actuator_ids = np.array([self.model.actuator(f'actuator{i}').id for i in range(1,len(joint_names)+1)])

        self.controller = 'osc'
        self.initial_pose = initial_pose
        self.control_timestep = 1 / control_freq
        fps = 30
        #steps / frame = seconds / frame * steps / second
        self.steps_per_frame = int(1 / (fps * self.model.opt.timestep))
        self.current_timestep = 0
        self.frames = []

    def reset_model(self):
        #maybe should be in reset instead
        self.current_timestep = 0

        key_id = self.model.key(self.initial_pose).id

        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        mujoco.mj_forward(self.model, self.data)

        pL = self.data.site_xpos[self.model.site("left_tip").id]
        pR = self.data.site_xpos[self.model.site("right_tip").id]

        #gripper opening width when fully open
        self.gripper_max = np.linalg.norm(pL - pR, 2)
        #gripper opening width when fully closed
        self.gripper_min = .0035 #have to edit if change site locations

        return self._get_obs()

    def _set_action_space(self):
        self.action_space = spaces.Box(
            low=np.array([-np.inf]*6 + [-1]),
            high=np.array([np.inf]*6 + [1]), dtype=np.float32)
        return self.action_space
    
    def _get_obs(self):
        #only valid since in this case panda robot base frame = world frame
        #o.w. need to go from world -> base first
        # 3x3 rotation matrix
        Rmat = self.data.site(self.ee_site_id).xmat.reshape(3, 3)
        # Convert to Euler angles
        site_euler = R.from_matrix(Rmat).as_euler("xyz", degrees=True) #degrees=False

        pL = self.data.site_xpos[self.model.site("left_tip").id]
        pR = self.data.site_xpos[self.model.site("right_tip").id]

        #fully open = 0, np.linalg.norm(pL - pR, 2) = gripper_max
        #fully closed = 1, np.linalg.norm(pL - pR, 2) = gripper_min
        gripper_state = 1 - (np.linalg.norm(pL - pR, 2) - self.gripper_min) / (self.gripper_max - self.gripper_min)
        gripper_state = np.clip(gripper_state, 0, 1)

        return self.data.site(self.ee_site_id).xpos.copy(), site_euler, np.array([gripper_state])
    
    def delta_to_target(self, delta):
        #converts action delta to a fixed target to step
        #target will be kept fixed for the action repeat for both pose and gripper
        delta_xyz = delta[0:3]
        delta_ori = delta[3:6] #currently ignore orientation
        delta_gripper = delta[6]

        current_ee_pos = self.data.site(self.ee_site_id).xpos.copy()
        gripper_ctrlrange = self.model.actuator_ctrlrange[-1,:]
        #255 = max close, 0 = max open
        # a_grip ∈ [-1, 1]
        # +a_grip = close, -a_grip = open
        gripper_delta_ctrl = (gripper_ctrlrange[1] - gripper_ctrlrange[0]) * delta_gripper + gripper_ctrlrange[0]
        gripper_ctrl = self.data.ctrl.copy()[self.model.actuator('fingers_actuator').id] + gripper_delta_ctrl

        target = np.empty(7)
        target[0:3] = current_ee_pos + delta_xyz
        target[6] = np.clip(gripper_ctrl, *gripper_ctrlrange)

        return target

    def step(self, target):
        target_xyz = target[0:3]
        target_gripper = target[6]
        
        self.data.ctrl[self.model.actuator('fingers_actuator').id] = target_gripper

        #controller- should really create another class
        #adds up dq until target is achieved
        for i in range(int(self.control_timestep / self.model.opt.timestep)):
            if (self.current_timestep % self.steps_per_frame) == 0:
                frame = self.render()
                self.frames.append(frame)

            if (self.controller == 'osc'):
                tau = osc(self.model, self.data, target_xyz, 'home')
                self.data.ctrl[self.actuator_ids] = tau[self.actuator_ids]

            #sim only advances during controller loop, so it's not running during long inference times
            mujoco.mj_step(self.model, self.data)
            self.current_timestep += 1

        observation = self._get_obs()
        reward = 0
        info = None

        if self.render_mode == "human":
            self.render()

        # Get geom ID for the cube
        box_id = mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")

        # World position of the cube
        box_pos = self.data.xpos[box_id]

        # z-coordinate = height above origin
        box_height = box_pos[2]

        if box_height > 0.2:
            return observation, 1, True, False, info

        #not sure if this is true if not using make (registered envs)
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, False, False, info

def main():
    env = PickPlacePandaEnvController(
        "C:\\Users\\gdev\\Documents\\CS\\DL\\projects\\Robotics\\custom_gym_env\\robots/franka_emika_panda/pick_place_custom.xml",
        render_mode="human")

if __name__ == "__main__":
    main()