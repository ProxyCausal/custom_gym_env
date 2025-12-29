import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

import numpy as np
import mujoco
from mujoco import mj_name2id

from scipy.spatial.transform import Rotation as R

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
            spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),   # ee_quat
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
        site_name = "attachment_site"
        site_id = self.model.site(site_name).id

        #only valid since in this case panda robot base frame = world frame
        #o.w. need to go from world -> base first
        # 3x3 rotation matrix
        Rmat = self.data.site(site_id).xmat.reshape(3, 3)
        # Convert to Euler angles
        site_euler = R.from_matrix(Rmat).as_euler("xyz", degrees=True)

        return self.data.site(site_id).xpos.copy(), site_euler, np.array([self.data.qpos[7]])

        #return self.data.qpos

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