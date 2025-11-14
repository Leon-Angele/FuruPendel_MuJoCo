"""TD3 training and evaluation script for the Steval rotary inverted pendulum.

All training / evaluation hyperparameters are defined at the top of this file
so you can easily edit them. `main()` only accepts a single argument: the
mode, either `train` or `eval`.

Note: This script assumes `stevalPendelEnv.py` is in the same folder and provides
the `StevalPendelEnv` class returning observations [cos(theta), sin(theta), theta_dot].
"""

# ---------------------- Hyperparameters / Defaults ----------------------
# Edit these constants to change training / evaluation behaviour.
TOTAL_TIMESTEPS = 100_000
LEARNING_RATE = 1e-3
BUFFER_SIZE = 1_000_000
BATCH_SIZE = 100
TAU = 0.005
GAMMA = 0.99
TRAIN_FREQ = 1
GRADIENT_STEPS = 1
LEARNING_STARTS = 100
POLICY = 'MlpPolicy'
# Set POLICY_KWARGS to a dict, e.g. {"net_arch": [64,64]} or None
POLICY_KWARGS = None
ACTION_NOISE_STD = 0.2
TENSORBOARD_LOG = "./logs"
SAVE_PATH = "td3_steval.zip"
MODEL_PATH = "td3_steval.zip"
EVAL_EPISODES = 5
RENDER = False
RENDER_PAUSE = 0.02
DEVICE = "auto"
NUM_ENVS = 12
SEED = 0
# -----------------------------------------------------------------------

import argparse
import json
import math
import os
import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common import logger as sb3_logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import logging

from stevalPendelEnv import StevalPendelEnv


def make_env(seed: Optional[int] = None):
    return lambda: StevalPendelEnv()


def get_action_noise(env, sigma: float):
    # TD3 expects an action noise object; for single-dim action use NormalActionNoise
    n_actions = env.action_space.shape[-1]
    return NormalActionNoise(mean=np.zeros(n_actions), sigma=sigma * np.ones(n_actions))


def simple_render(obs, ax, line):
    # obs: [cos(theta), sin(theta), theta_dot]
    cos_theta, sin_theta = float(obs[0]), float(obs[1])
    theta = math.atan2(sin_theta, cos_theta)
    x = math.sin(theta)
    y = math.cos(theta)
    line.set_data([0.0, x], [0.0, y])
    return


def setup_matplotlib():
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    line, = ax.plot([0, 0], [0, 1], marker='o', lw=4)
    return fig, ax, line


def train():
    if TENSORBOARD_LOG:
        os.makedirs(TENSORBOARD_LOG, exist_ok=True)

    # Configure sb3 logger to include tensorboard if possible, otherwise ensure stdout
    try:
        sb3_logger.configure(folder=TENSORBOARD_LOG, format_strs=['stdout', 'tensorboard'])
    except TypeError:
        # signature may not accept format_strs; fall back
        try:
            sb3_logger.configure(folder=TENSORBOARD_LOG)
        except Exception:
            pass

    # Ensure Python logging prints INFO to stdout
    root_logger = logging.getLogger()
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        sh.setFormatter(formatter)
        root_logger.addHandler(sh)
    root_logger.setLevel(logging.INFO)

    # Create NUM_ENVS Monitor-wrapped envs that write monitor files into the tensorboard folder
    env_fns = []
    for i in range(NUM_ENVS):
        def make_fn(i):
            def _init():
                e = StevalPendelEnv()
                monitor_path = os.path.join(TENSORBOARD_LOG, f"monitor_{i}.csv")
                return Monitor(e, filename=monitor_path)
            return _init
        env_fns.append(make_fn(i))

    env = DummyVecEnv(env_fns)

    # Build action noise if requested
    action_noise = None
    if ACTION_NOISE_STD and ACTION_NOISE_STD > 0.0:
        single_env = StevalPendelEnv()
        action_noise = get_action_noise(single_env, ACTION_NOISE_STD)

    model = TD3(
        policy=POLICY,
        env=env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        tau=TAU,
        gamma=GAMMA,
        train_freq=TRAIN_FREQ,
        gradient_steps=GRADIENT_STEPS,
        learning_starts=LEARNING_STARTS,
        action_noise=action_noise,
        policy_kwargs=POLICY_KWARGS,
        verbose=2,
        tensorboard_log=TENSORBOARD_LOG,
        seed=SEED,
        device=DEVICE,
    )

    print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
    # Callback prints episode summaries from Monitor and records to sb3 logger (also to tensorboard)
    class EpisodeLoggerCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)

        def _on_step(self) -> bool:
            infos = self.locals.get('infos')
            if infos:
                for info in infos:
                    if info and 'episode' in info:
                        ep = info['episode']
                        # Print a concise line for each finished episode
                        print(f"Episode finished - reward: {ep['r']:.3f}, length: {ep['l']}")
                        try:
                            sb3_logger.record('episode_reward', float(ep['r']))
                            sb3_logger.record('episode_length', int(ep['l']))
                            sb3_logger.dump(step=self.num_timesteps)
                        except Exception:
                            pass
            return True

    callback = EpisodeLoggerCallback()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True, callback=callback)

    save_path = SAVE_PATH
    model.save(save_path)
    print(f"Model saved to: {save_path}")
    env.close()


def evaluate():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    # Create environment with human render mode so env.render() uses MuJoCo viewer
    env = StevalPendelEnv(render_mode='human')

    # Load model (no need to pass env to load)
    model = TD3.load(MODEL_PATH)

    all_rewards = []
    for ep in range(EVAL_EPISODES):
        obs, info = env.reset()
        terminated = False
        truncated = False
        ep_reward = 0.0
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            # Render using env.render(); StevalPendelEnv will try the MuJoCo viewer first
            try:
                env.render()
            except Exception:
                # ignore render failures during evaluation
                pass
            if terminated or truncated:
                break
        all_rewards.append(ep_reward)
        print(f"Episode {ep+1}/{EVAL_EPISODES} reward: {ep_reward}")

    env.close()
    print(f"Average reward over {EVAL_EPISODES} episodes: {np.mean(all_rewards)}")


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate TD3 on the Steval Pendulum environment.")
    parser.add_argument("mode", choices=["train", "eval"], help="Operation mode: train or eval")

    args = parser.parse_args()
    if args.mode == 'train':
        train()
    else:
        evaluate()


if __name__ == '__main__':
    main()
