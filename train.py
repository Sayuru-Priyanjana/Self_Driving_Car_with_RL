"""
train.py — Train the PPO agent on the self-driving car environment.

Usage:
    python train.py                          # default 200 000 timesteps
    python train.py --timesteps 500000
    python train.py --timesteps 200000 --render   # watch training (slow)
"""

import glob
import os
import sys
import argparse
from typing import Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util   import make_vec_env
from stable_baselines3.common.callbacks  import (
    BaseCallback, EvalCallback, CheckpointCallback
)
from stable_baselines3.common.monitor    import Monitor

# make sure local modules are importable when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from env   import CarEnv
from utils import plot_rewards, plot_training_progress, RewardLogger


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class RewardLoggerCallback(BaseCallback):
    """Collects per-episode rewards from the Monitor wrapper and logs them."""

    def __init__(self, logger: RewardLogger, verbose=0):
        super().__init__(verbose)
        self.rl  = logger
        self._ep = 0
        self._ep_rewards: list = []
        self._ep_lengths: list = []

    def _on_step(self) -> bool:
        # SB3 populates infos with Monitor episode data
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep_r = info["episode"]["r"]
                ep_l = info["episode"]["l"]
                laps = info.get("laps", 0)
                self._ep += 1
                self._ep_rewards.append(ep_r)
                self._ep_lengths.append(ep_l)
                self.rl.log(self._ep, ep_r, ep_l, laps)
                if self._ep % 10 == 0:
                    mean_r = np.mean(self._ep_rewards[-20:])
                    print(f"  ep {self._ep:>4d}  "
                          f"reward {ep_r:+8.1f}  "
                          f"mean20 {mean_r:+8.1f}  "
                          f"steps {ep_l}")
        return True

    def get_episode_rewards(self):
        return self._ep_rewards


class ProgressPlotCallback(BaseCallback):
    """Saves a reward-curve plot every N timesteps."""

    def __init__(self, reward_cb: RewardLoggerCallback,
                 plot_every: int = 20_000, verbose=0):
        super().__init__(verbose)
        self._rcb      = reward_cb
        self._plot_every = plot_every

    def _on_step(self) -> bool:
        if self.num_timesteps % self._plot_every == 0:
            rewards = self._rcb.get_episode_rewards()
            if rewards:
                plot_rewards(rewards, save_path="plots/reward_curve.png")
        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train PPO self-driving car")
    p.add_argument("--timesteps", type=int,   default=200_000,
                   help="Total environment steps (default: 200 000)")
    p.add_argument("--n-envs",    type=int,   default=1,
                   help="Parallel environments (default: 1)")
    p.add_argument("--render",    action="store_true",
                   help="Render the environment during training (slow)")
    p.add_argument("--resume",    type=str,   default=None,
                   help="Path to existing model zip to continue training")
    p.add_argument("--resume-latest", action="store_true",
                   help="Resume training from the latest saved model or checkpoint")
    p.add_argument("--random-spawn", action="store_true",
                   help="Randomise car spawn position each episode")
    return p.parse_args()


def make_env(render: bool = False, random_spawn: bool = False):
    """Factory that returns a Monitor-wrapped CarEnv."""
    def _init():
        env = CarEnv(
            render_mode    = "human" if render else None,
            max_steps      = 2000,
            randomise_spawn= random_spawn,
        )
        env = Monitor(env, filename=None)
        return env
    return _init


def find_latest_checkpoint() -> Optional[str]:
    candidates = []
    model_file = "models/ppo_car.zip"
    if os.path.exists(model_file):
        candidates.append(model_file)
    candidates.extend(glob.glob("models/checkpoints/ppo_car_*_steps.zip"))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def main():
    args = parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs",   exist_ok=True)
    os.makedirs("plots",  exist_ok=True)

    print("=" * 55)
    print("  PPO Self-Driving Car — Training")
    print("=" * 55)
    print(f"  Total timesteps : {args.timesteps:,}")
    print(f"  Parallel envs   : {args.n_envs}")
    print(f"  Render          : {args.render}")
    print(f"  Random spawn    : {args.random_spawn}")
    resume_path = args.resume
    if args.resume_latest and resume_path is None:
        resume_path = find_latest_checkpoint()
        if resume_path:
            print(f"  Resuming from latest checkpoint: {resume_path}")

    if resume_path:
        print(f"  Resuming from   : {resume_path}")
    print("=" * 55)

    # ---- vectorised env ----
    env = make_vec_env(
        make_env(render=args.render, random_spawn=args.random_spawn),
        n_envs=args.n_envs,
    )

    # ---- model ----
    if resume_path:
        print(f"\n[train] Loading model from {resume_path} ...")
        model = PPO.load(resume_path, env=env)
    else:
        model = PPO(
            policy         = "MlpPolicy",
            env            = env,
            learning_rate  = 3e-4,
            n_steps        = 2048,
            batch_size     = 64,
            n_epochs       = 10,
            gamma          = 0.99,
            gae_lambda     = 0.95,
            clip_range     = 0.2,
            ent_coef       = 0.01,
            vf_coef        = 0.5,
            max_grad_norm  = 0.5,
            verbose        = 0,
            device         = "cpu",
        )

    # ---- callbacks ----
    reward_logger   = RewardLogger("logs/rewards.csv")
    reward_cb       = RewardLoggerCallback(reward_logger)
    progress_plot_cb= ProgressPlotCallback(reward_cb, plot_every=20_000)

    checkpoint_cb = CheckpointCallback(
        save_freq      = 50_000,
        save_path      = "models/checkpoints/",
        name_prefix    = "ppo_car",
        verbose        = 1,
    )

    callbacks = [reward_cb, progress_plot_cb, checkpoint_cb]

    # ---- train ----
    print("\n[train] Starting training …\n")
    model.learn(
        total_timesteps     = args.timesteps,
        callback            = callbacks,
        reset_num_timesteps = not bool(resume_path),
    )

    # ---- save ----
    model.save("models/ppo_car")
    print("\n[train] Model saved → models/ppo_car.zip")

    # ---- final plots ----
    ep_rewards = reward_cb.get_episode_rewards()
    if ep_rewards:
        plot_rewards(ep_rewards, save_path="plots/reward_curve.png")
        print("[train] Reward plot saved → plots/reward_curve.png")

    env.close()
    print("\n[train] Done!")


if __name__ == "__main__":
    main()