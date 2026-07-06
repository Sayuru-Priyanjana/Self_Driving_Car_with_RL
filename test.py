"""
test.py — Load a trained PPO model and run it with full rendering.

Usage:
    python test.py                          # load models/ppo_car.zip
    python test.py --model models/ppo_car.zip
    python test.py --episodes 5 --slow-mo
    python test.py --random-spawn
"""

import os
import sys
import time
import argparse
import numpy as np
import pygame

sys.path.insert(0, os.path.dirname(__file__))

from stable_baselines3 import PPO
from env   import CarEnv
from utils import plot_rewards, RewardLogger


def parse_args():
    p = argparse.ArgumentParser(description="Test trained PPO car agent")
    p.add_argument("--model",        type=str, default="models/ppo_car.zip")
    p.add_argument("--episodes",     type=int, default=5)
    p.add_argument("--slow-mo",      action="store_true",
                   help="Run at 15 fps instead of 60")
    p.add_argument("--random-spawn", action="store_true")
    p.add_argument("--no-sensors",   action="store_true",
                   help="Hide sensor rays")
    p.add_argument("--deterministic",action="store_true",
                   help="Use deterministic policy actions")
    return p.parse_args()


def run_episode(model, env, deterministic: bool = True) -> dict:
    """Run one episode; return stats dict."""
    obs, _ = env.reset()
    total_r = 0.0
    steps   = 0
    laps    = 0
    done    = False

    while not done:
        # handle window close
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                sys.exit(0)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    env.close()
                    sys.exit(0)

        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_r += reward
        steps   += 1
        laps     = info.get("laps", 0)
        done     = terminated or truncated

    return {"total_reward": total_r, "steps": steps, "laps": laps}


def main():
    args = parse_args()

    if not os.path.exists(args.model):
        print(f"[test] ERROR: Model file not found: {args.model}")
        print("[test] Train first with:  python train.py")
        sys.exit(1)

    print("=" * 50)
    print("  PPO Self-Driving Car — Testing")
    print("=" * 50)
    print(f"  Model    : {args.model}")
    print(f"  Episodes : {args.episodes}")
    print(f"  Slow-mo  : {args.slow_mo}")
    print("=" * 50)
    print("\n[test] Press ESC or close the window to quit.\n")

    # load model
    model = PPO.load(args.model, device="cpu")

    # build env with rendering
    env = CarEnv(
        render_mode     = "human",
        max_steps       = 3000,
        randomise_spawn = args.random_spawn,
        slow_mo         = args.slow_mo,
        show_sensors    = not args.no_sensors,
    )

    all_rewards = []
    for ep in range(1, args.episodes + 1):
        stats = run_episode(
            model, env,
            deterministic = args.deterministic,
        )
        all_rewards.append(stats["total_reward"])
        print(f"  Episode {ep:>2d}:  "
              f"reward={stats['total_reward']:+8.1f}  "
              f"steps={stats['steps']:>4d}  "
              f"laps={stats['laps']}")

    env.close()

    print("\n[test] Summary:")
    print(f"  Mean reward : {np.mean(all_rewards):+.1f}")
    print(f"  Best episode: {max(all_rewards):+.1f}")
    print(f"  Worst episode:{min(all_rewards):+.1f}")

    if all_rewards:
        os.makedirs("plots", exist_ok=True)
        plot_rewards(all_rewards, window=min(5, len(all_rewards)),
                     save_path="plots/test_rewards.png")
        print("[test] Test-reward plot saved → plots/test_rewards.png")


if __name__ == "__main__":
    main()
