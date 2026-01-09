import mujoco
import robosuite
from robosuite.controllers import load_composite_controller_config

env = robosuite.make(
    env_name="Lift",
    robots="Panda",
    controller_configs=load_composite_controller_config(robot='Panda'),
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

# Get the MuJoCo model
model = env.sim.model._model  # low-level mujoco.MjModel

# Dump the fully resolved XML
#mujoco.mj_saveLastXML("outputs/robosuite_final.xml", model)

print(model.opt.timestep)
print(model.opt.integrator)
print(model.opt.iterations)
print(model.opt.noslip_iterations)
print(model.opt.cone)