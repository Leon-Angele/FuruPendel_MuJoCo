"""TD3 training and evaluation script for the Steval rotary inverted pendulum.

All training / evaluation hyperparameters are defined at the top of this file
so you can easily edit them. `main()` only accepts a single argument: the
mode, either `train` or `eval`.

Note: This script assumes `stevalPendelEnv.py` is in the same folder and provides
the `StevalPendelEnv` class returning observations [cos(theta), sin(theta), theta_dot].
"""

# ---------------------- Hyperparameters / Defaults ----------------------
# Edit these constants to change training / evaluation behaviour.
TOTAL_TIMESTEPS = 200_000
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
ACTION_NOISE_STD = 0.1
TENSORBOARD_LOG = None
SAVE_PATH = "td3_steval.zip"
MODEL_PATH = "td3_steval.zip"
EVAL_EPISODES = 5
RENDER = False
RENDER_PAUSE = 0.02
DEVICE = "auto"
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
from stable_baselines3.common.noise import NormalActionNoise

from stevalPendelEnv import StevalPendelEnv


def make_env(render_mode: Optional[str] = None, seed: Optional[int] = None):
    return lambda: StevalPendelEnv(render_mode=render_mode)


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

    # Create single environment instance
    env = StevalPendelEnv(render_mode=None)

    # Build action noise if requested
    action_noise = None
    if ACTION_NOISE_STD and ACTION_NOISE_STD > 0.0:
        action_noise = get_action_noise(env, ACTION_NOISE_STD)

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
        verbose=1,
        tensorboard_log=TENSORBOARD_LOG,
        seed=SEED,
        device=DEVICE,
    )

    print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    save_path = SAVE_PATH
    model.save(save_path)
    print(f"Model saved to: {save_path}")
    env.close()


def evaluate():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    # Create env with human rendering if RENDER True so env.render() uses MuJoCo viewer
    env = StevalPendelEnv(render_mode='human' if RENDER else None)

    # Load model and set env
    model = TD3.load(MODEL_PATH)
    model.set_env(env)

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
            if RENDER:
                # Use env.render() which will prefer MuJoCo viewer and fallback to Matplotlib
                try:
                    env.render()
                except Exception:
                    # If render crashes, fallback to the simple renderer
                    simple_render(obs, None, None)
                    plt.pause(RENDER_PAUSE)
            if terminated or truncated:
                break
        all_rewards.append(ep_reward)
        print(f"Episode {ep+1}/{EVAL_EPISODES} reward: {ep_reward}")

    env.close()
    if RENDER:
        plt.ioff()
        plt.show()

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
