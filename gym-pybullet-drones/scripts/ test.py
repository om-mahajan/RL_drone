import pandas as pd
import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch as th
from gym_pybullet_drones.envs.Aviary import Aviary
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
import wandb
from wandb.integration.sb3 import WandbCallback

def run(multiagent=False, output_folder='Results', gui=True, plot=True, colab=False, record_video=True, local=True):

    train_env = make_vec_env(Aviary,
                            env_kwargs=dict(obs=ObservationType('kin'), act=ActionType('rpm')),
                            n_envs=1,
                            seed=1
                            )
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    eval_env = DummyVecEnv([lambda: Aviary(obs=ObservationType('kin'), act=ActionType('rpm'))])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # Set evaluation env to not update stats
    eval_env.training = False
    eval_env.norm_reward = False

    learning_rate = 1e-4
    clip_range = 0.2
    ent_coef = 0.01
    n_steps = 1024
    target_kl = 0.02
    batch_size = 512
    n_epochs = 4
    gamma = 0.99
    policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=[128, 128])
    Model = PPO("MlpPolicy", train_env,learning_rate=learning_rate,n_steps=n_steps,batch_size=batch_size,n_epochs=n_epochs,gamma=gamma,clip_range=clip_range,target_kl=target_kl,policy_kwargs=policy_kwargs,verbose=1,tensorboard_log="./tb_logs/")
    target_reward = 467
    
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward,
                                                     verbose=1)
    obs= train_env.reset()
    print(np.shape(obs), 'obs shape')
    print(obs,'obs')
    obs, reward, terminated, truncated, info = train_env.step([1,1,0,0])
    print(reward,np.shape(obs))
    


if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--multiagent',         default=False,            type=str2bool,      help='Whether to use example LeaderFollower instead of Hover (default: False)', metavar='')
    parser.add_argument('--gui',                default=True,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=True,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default='Results', type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=False,         type=bool,          help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))