import time

import numpy as np
from geometry_msgs.msg import PoseStamped
from stable_baselines3 import PPO
import torch as th
from mujocoEnv import mujocoEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd, to_absolute_path
import os
import rospy

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    global can_run, use_wind, env
    print(OmegaConf.to_yaml(cfg))
    # create output directory
    prefix_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    # set root directory
    os.chdir(prefix_path)  
    # create environment
    env = mujocoEnv(cfg.env)
    env = Monitor(env, info_keywords=("is_success",))
    env = DummyVecEnv([lambda: env])

    model = PPO.load(to_absolute_path(cfg.network.load_path), env=env)

    obs = env.reset()
    # evaluate
    while True:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        if done:
            break



if __name__ == "__main__":
    main()
