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

class PickPlacePandaEnv(MujocoEnv):
    def __init__(
        self,
        xml_file: str = "scene.xml",
        frame_skip: int = 5,
        default_camera_config: dict[str, float | int] = None, #DEFAULT_CAMERA_CONFIG
        **kwargs
    ):
        #action space is set automatically by MujocoEnv
        #6 from robot, 7 from free joint (3 xyz, 4 quat)
        #is this correct for panda?
        #observation_space = Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float64)

        observation_space = spaces.Tuple((
            spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),   # ee_pos
            spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),   # ee_euler
            spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)  # gripper qpos
        ))

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )

    def reset_model(self):
        key_id = self.model.key('home').id

        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)

        mujoco.mj_forward(self.model, self.data)

        return self._get_obs()

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

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
    
    def _get_obs(self):
        #only valid since in this case panda robot base frame = world frame
        #o.w. need to go from world -> base first
        # 3x3 rotation matrix
        Rmat = self.data.site(self.ee_site_id).xmat.reshape(3, 3)
        # Convert to Euler angles
        site_euler = R.from_matrix(Rmat).as_euler("xyz", degrees=True)

        return self.data.site(self.ee_site_id).xpos.copy(), site_euler, np.array([self.data.qpos[7]])

        #return self.data.qpos

class PickPlacePandaEnvController(MujocoEnv):
    def __init__(
        self,
        xml_file: str = "scene.xml",
        frame_skip: int = 5, #shouldn't be relevant since we're not going to be controlling the timesteps manually and not via do_simulation
        default_camera_config: dict[str, float | int] = None, #DEFAULT_CAMERA_CONFIG
        **kwargs
    ):
        observation_space = spaces.Tuple((
            spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),   # ee_pos
            spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),   # ee_euler
            spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)  # gripper qpos
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

    def reset_model(self):
        key_id = self.model.key('home').id

        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)

        mujoco.mj_forward(self.model, self.data)

        return self._get_obs()

    def _set_action_space(self):
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32)
        return self.action_space
    
    def _get_obs(self):
        #only valid since in this case panda robot base frame = world frame
        #o.w. need to go from world -> base first
        # 3x3 rotation matrix
        Rmat = self.data.site(self.ee_site_id).xmat.reshape(3, 3)
        # Convert to Euler angles
        site_euler = R.from_matrix(Rmat).as_euler("xyz", degrees=True)

        return self.data.site(self.ee_site_id).xpos.copy(), site_euler, np.array([self.data.qpos[7]])

    def step(self, action):
        delta_xyz = action[0:3]
        delta_ori = action[3:6]
        gripper_state = action[6]
        
        #technically need to convert extrinsic Euler angles rel to base deltas into quarternion deltas
        #but gpt says it's ok if the deltas are small
        
        #delta = diff b/w desired - current pos = actions = errors for xyz, but not angles?
        
        #action is ctrl- for gripper state this != joint angles

        current_ee_pos = self.data.site(self.ee_site_id).xpos.copy()
        desired_ee_pos = current_ee_pos + delta_xyz

        #controller- should really create another class
        #adds up dq until target is achieved
        iters = 0
        max_steps = 1000
        print("Received action, controller is trying to achieve target position")
        while np.linalg.norm(current_ee_pos - desired_ee_pos, 2) > .001: #.01
            if (iters > max_steps):
                print(f"Controllers ran {max_steps}, moving on to next action")
                break

            if (iters % 100) == 0:
                print(f"Controller error: {current_ee_pos - desired_ee_pos}, l2 dist: {np.linalg.norm(current_ee_pos - desired_ee_pos, 2)}")
            
            if (self.controller == 'diffik'):
                integration_dt: float = 0.1
                dq = diffik_nullspace(self.model, self.data, desired_ee_pos)
                q = self.data.qpos.copy()  # Note the copy here is important.
                mujoco.mj_integratePos(self.model, q, dq, integration_dt)
                np.clip(q[self.dof_ids], *self.model.jnt_range.T[:,:len(self.dof_ids)], out=q[self.dof_ids])
                # Set the control signal and step the simulation.
                self.data.ctrl[self.actuator_ids] = q[self.dof_ids]
                
            elif (self.controller == 'osc'):
                tau = osc(self.model, self.data, desired_ee_pos)
                self.data.ctrl[self.actuator_ids] = tau[self.actuator_ids]

            mujoco.mj_step(self.model, self.data)

            current_ee_pos = self.data.site(self.ee_site_id).xpos.copy()

            iters += 1

        gripper_ctrlrange = self.model.actuator_ctrlrange[-1,:]
        gripper_unscaled = (gripper_ctrlrange[1] - gripper_ctrlrange[0]) * gripper_state + gripper_ctrlrange[0]
        self.data.ctrl[self.model.actuator('actuator8').id] = gripper_unscaled

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
    #env = PickPlaceSO101Env("scene.xml", render_mode="human")
    env = PickPlacePandaEnv(
        "C:\\Users\\gdev\\Documents\\CS\\DL\\projects\\Robotics\\custom_gym_env\\robots/franka_emika_panda/pick_place_custom.xml",
        render_mode="rgb_array", camera_name='fixed_cam')

    #print(f"Action space {env.action_space}")

    obs, info = env.reset()
    print(obs)
    steps = 0
    while True:
        action = env.action_space.sample()               # your control logic here
        #action_xyz = VJEPA(obs)
        #action = IK(action_xyz)
        obs, reward, terminated, truncated, info = env.step(action)

        if steps == 500:
            from PIL import Image
            #Image.fromarray(env.render()).save("fixed_cam.png")

        if terminated or truncated:
            break

        steps += 1

if __name__ == "__main__":
    main()