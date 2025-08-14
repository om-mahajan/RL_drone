import os
import time
import numpy as np
from stable_baselines3 import PPO
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync

# === CONFIG ===
MODEL_PATH = "/home/om/drone/gym-pybullet-drones/gym_pybullet_drones/examples/results/save-08.09.2025_07.49.19/final_model"  # path to your trained model
MULTIAGENT = False  # change to True if you trained MultiHoverAviary
NUM_DRONES = 1
GUI = True 
RECORD_VIDEO = True

# === CREATE ENVIRONMENT ===

env = HoverAviary(gui=GUI,
                        obs=ObservationType.KIN,
                        act=ActionType.ONE_D_RPM,
                        record=True)


# === LOAD MODEL ===
model = PPO.load(MODEL_PATH)

# === RESET ENV ===
obs, info = env.reset(seed=42, options={})
print(np.shape(obs), "---------------------------------------- obs")

start = time.time()
for i in range((env.EPISODE_LEN_SEC+2)*env.CTRL_FREQ):
    # Modify observation to include target info if your policy was trained that way
    # Example: Append relative target position
    # obs_with_target = np.concatenate([obs, target_position - obs[0:3]])

    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    print("---------------action-----------", action, "--------------action------------")
    print("----------------------------Reward--------------",reward, "----------------------------Reward--------------")
    env.render()
    sync(i, start, env.CTRL_TIMESTEP)

    if terminated:
        obs, info = env.reset(seed=42, options={})

env.close()
