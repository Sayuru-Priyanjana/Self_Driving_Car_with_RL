"""
utils.py — Helper utilities: reward plotting, HUD rendering, logging.
"""

import os
import math
import numpy as np
import pygame
import matplotlib
matplotlib.use("Agg")          # non-interactive backend; safe on headless machines
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_rewards(episode_rewards: list,
                 window: int = 50,
                 save_path: str = "plots/reward_curve.png",
                 show: bool = False):
    """
    Plot raw episode rewards + moving-average smoothing.
    Saves to `save_path` and optionally shows the window.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    rewards = np.array(episode_rewards, dtype=float)
    episodes = np.arange(1, len(rewards) + 1)

    # moving average
    if len(rewards) >= window:
        kernel = np.ones(window) / window
        smooth = np.convolve(rewards, kernel, mode="valid")
        smooth_x = episodes[window - 1:]
    else:
        smooth = rewards
        smooth_x = episodes

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, rewards, color="#4a90d9", alpha=0.35, linewidth=1,
            label="Episode reward")
    ax.plot(smooth_x, smooth, color="#e8534a", linewidth=2,
            label=f"{window}-ep moving avg")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Total Reward", fontsize=12)
    ax.set_title("PPO Self-Driving Car — Training Rewards", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    if show:
        plt.show()
    print(f"[utils] Reward plot saved → {save_path}")


def plot_training_progress(timesteps: list,
                            mean_rewards: list,
                            save_path: str = "plots/training_progress.png"):
    """Plot SB3 evaluation mean rewards versus total timesteps."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(timesteps, mean_rewards, "o-", color="#2ecc71", linewidth=2,
            markersize=5)
    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Mean Episode Reward", fontsize=12)
    ax.set_title("PPO Training Progress", fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"[utils] Progress plot saved → {save_path}")


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------

class HUD:
    """Overlay that draws real-time telemetry on the Pygame window."""

    def __init__(self, font_size: int = 18):
        pygame.font.init()
        self.font       = pygame.font.SysFont("monospace", font_size)
        self.small_font = pygame.font.SysFont("monospace", font_size - 4)
        self.panel_w    = 200
        self.panel_h    = 160
        self.padding    = 8

    def draw(self, surface: pygame.Surface,
             speed:   float,
             reward:  float,
             total_r: float,
             step:    int,
             episode: int,
             laps:    int,
             steering: float):

        # semi-transparent background panel
        panel = pygame.Surface((self.panel_w, self.panel_h), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 160))
        surface.blit(panel, (6, 6))

        lines = [
            f"Episode : {episode}",
            f"Step    : {step}",
            f"Speed   : {speed:.2f}",
            f"Steer   : {steering:+.2f}",
            f"Reward  : {reward:+.2f}",
            f"Total R : {total_r:.1f}",
            f"Laps    : {laps}",
        ]
        colours = [
            (200, 200, 200),
            (200, 200, 200),
            (100, 220, 100),
            (220, 180, 100),
            (reward >= 0) and (100, 255, 100) or (255, 100, 100),
            (180, 180, 255),
            (255, 215, 0),
        ]
        for idx, (text, col) in enumerate(zip(lines, colours)):
            surf = self.font.render(text, True, col)
            surface.blit(surf, (self.padding + 6,
                                self.padding + 6 + idx * (self.font.get_height() + 2)))

    def draw_message(self, surface: pygame.Surface, msg: str,
                     colour=(255, 80, 80)):
        """Centre-screen overlay message (crash, lap, etc.)."""
        w, h = surface.get_size()
        text = self.font.render(msg, True, colour)
        rect = text.get_rect(center=(w // 2, h // 2))
        surface.blit(text, rect)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

class RewardLogger:
    """Simple CSV logger for episode rewards."""

    def __init__(self, path: str = "logs/rewards.csv"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        with open(path, "w") as f:
            f.write("episode,total_reward,steps,laps\n")

    def log(self, episode: int, total_reward: float, steps: int, laps: int):
        with open(self.path, "a") as f:
            f.write(f"{episode},{total_reward:.4f},{steps},{laps}\n")

    @staticmethod
    def load(path: str = "logs/rewards.csv"):
        import csv
        episodes, rewards = [], []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                episodes.append(int(row["episode"]))
                rewards.append(float(row["total_reward"]))
        return episodes, rewards


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def normalise_angle(a: float) -> float:
    """Wrap angle to [-π, π]."""
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a


def clamp(v, lo, hi):
    return max(lo, min(hi, v))
