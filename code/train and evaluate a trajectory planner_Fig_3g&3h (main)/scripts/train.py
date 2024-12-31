from stable_baselines3 import PPO
import torch as th
from mujocoEnv import mujocoEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
import hydra
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.vec_env import DummyVecEnv
from hydra.utils import get_original_cwd, to_absolute_path
import os



class TensorboardCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
    
    def _on_step(self) -> bool:
        # FIXME: can not access
        info = self.locals["infos"][0]
        if info["return_terminal_reward"]:
            self.logger.record("success", int(info["is_success"]))
            self.logger.record("rew_tp", info["rew_tp"])
            self.logger.record("rew_tv", info["rew_tv"])
        self.logger.record("rew_p", info["rew_p"])
        self.logger.record("rew_v", info["rew_v"])
        self.logger.record("rew_t", info["rew_t"])
        return True

@hydra.main(config_path="../config", config_name="config",version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # create output directory
    prefix_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    os.chdir(prefix_path) # set root directory
    os.mkdir("models") # save models
    os.mkdir("tensorboard") # tensorboard

    # create environment
    env = mujocoEnv(cfg.env)
    env = Monitor(env, info_keywords=("is_success",))
    env = DummyVecEnv([lambda: env])
    env.reset()

    # create agent
    # Custom actor (pi) and value function (vf) networks
    # of two layers of size 32 each with Relu activation function
    # Note: an extra linear layer will be added on top of the pi and the vf nets, respectively
    # create callbacks
    checkpoint_callback = CheckpointCallback(save_freq=cfg.network.save_freq, 
                                            save_path=prefix_path + cfg.network.save_path,
                                            name_prefix='rl_model')
    tensorboard_callback = TensorboardCallback()
    callback = CallbackList([checkpoint_callback, tensorboard_callback])

    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                        net_arch=dict(pi=cfg.network.pi, vf=cfg.network.vf))
    model = PPO(cfg.network.policy, env, verbose=cfg.network.verbose, batch_size=cfg.network.batch_size,
                n_steps=cfg.network.n_steps, seed=cfg.network.seed, policy_kwargs=policy_kwargs,
                    tensorboard_log=prefix_path + cfg.network.tensorboard_log)

    # whether to load model
    if cfg.network.if_load:
        # model = PPO.load(to_absolute_path(cfg.network.load_path), env=env)
        print("Load Model")
        model.set_parameters(to_absolute_path(cfg.network.load_path))

    model.learn(total_timesteps=cfg.network.max_episode * cfg.env.max_step, reset_num_timesteps=False, tb_log_name=cfg.network.name, callback=callback)
    model.save(prefix_path + cfg.network.save_path + cfg.network.name)

if __name__ == "__main__":
    main()