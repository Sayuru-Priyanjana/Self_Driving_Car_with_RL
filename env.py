"""
env.py — Custom Gymnasium environment for the PPO self-driving car.

Observation (5 floats, all in [0,1] or [-1,1]):
  [0] front sensor   (normalised 0-1)
  [1] left  sensor   (normalised 0-1)
  [2] right sensor   (normalised 0-1)
  [3] speed          (normalised 0-1)
  [4] angle to track centre-line (tanh-squashed, -1 to 1)

Action (1 continuous float in [-1, 1]):
  steering (-1 = full left, +1 = full right)

Reward:
  +1.0  proportional to centre-line proximity (max at centre)
  +0.2  proportional to forward speed
  +5.0  checkpoint passed
  -10.0 collision / off-road
  -0.01 per step (time penalty)
"""

import math
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import pygame

from car   import Car
from track import Track
from utils import HUD


class CarEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    # Reward weights
    W_CENTER     =  1.0
    W_FORWARD    =  0.5
    W_CHECKPOINT =  5.0
    W_CRASH      = -10.0
    W_TIME       = -0.01

    def __init__(self,
                 render_mode:     str  = None,
                 max_steps:       int  = 2000,
                 randomise_spawn: bool = False,
                 slow_mo:         bool = False,
                 show_sensors:    bool = True,
                 track_type:      str  = "oval"):

        super().__init__()
        self.render_mode     = render_mode
        self.max_steps       = max_steps
        self.randomise_spawn = randomise_spawn
        self.slow_mo         = slow_mo
        self.show_sensors    = show_sensors
        self.track_type      = track_type

        # Observation / action spaces
        low  = np.array([0, 0, 0, 0, -1], dtype=np.float32)
        high = np.array([1, 1, 1, 1,  1], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([ 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Track (shared; reset_checkpoints called each episode)
        self.track = Track(width=800, height=600, track_type=track_type)
        # centre-line reward saturates at half the actual road width
        self.max_center_dist = self.track.road_w / 2

        # Build the track surface immediately (headless — no display needed)
        self._ensure_headless_surface()

        self.car          = None
        self._screen      = None
        self._clock       = None
        self._hud         = None

        self._step         = 0
        self._episode      = 0
        self._total_reward = 0.0
        self._last_action  = 0.0

    # ------------------------------------------------------------------
    # Headless surface — allows collision checking without a display
    # ------------------------------------------------------------------
    def _ensure_headless_surface(self):
        """Build the track surface using an offscreen Surface."""
        if self.track._track_surf is not None:
            return
        if not pygame.get_init():
            pygame.init()
        # Build onto a plain Surface (no display required)
        self.track.build_surface()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.track.reset_checkpoints()
        sx, sy, sa = self.track.get_spawn(randomise=self.randomise_spawn)
        self.car = Car(sx, sy, sa)
        self.car.speed = 0.5   # small initial nudge so it isn't stationary

        self._step         = 0
        self._episode     += 1
        self._total_reward = 0.0
        self._last_action  = 0.0

        # Verify spawn is on road (warn if not)
        if not self.track.is_on_road(sx, sy):
            # fallback to absolute safe centre of top straight
            self.car.x = (self.track.center_pts[0][0] +
                          self.track.center_pts[len(self.track.center_pts)//4][0]) / 2
            self.car.y = self.track.center_pts[0][1]
            self.car.angle = 0.0

        obs  = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        steering = float(np.clip(action[0], -1.0, 1.0))
        self._last_action = steering

        self.car.step(steering, throttle=1.0)
        self._step += 1
        self.car.cast_sensors(self.track.wall_segments)

        # ---- reward ----
        reward = self.W_TIME

        # forward speed bonus
        reward += self.W_FORWARD * (self.car.speed / Car.MAX_SPEED)

        # centre-line proximity
        dist_c = self.track.dist_to_center(self.car.x, self.car.y)
        centre_ratio = max(0.0, 1.0 - dist_c / self.max_center_dist)
        reward += self.W_CENTER * centre_ratio

        # checkpoint
        if self.track.check_checkpoint(self.car.x, self.car.y):
            reward += self.W_CHECKPOINT

        # ---- collision ----
        terminated = False
        if self._is_crashed():
            reward    += self.W_CRASH
            terminated = True

        truncated = self._step >= self.max_steps
        self._total_reward += reward

        obs  = self._get_obs()
        info = {
            "laps":         self.track.lap_count,
            "total_reward": self._total_reward,
            "speed":        self.car.speed,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self._screen is None:
            self._init_display()

        self._screen.fill((34, 139, 34))
        self.track.draw(self._screen)
        self.car.draw(self._screen, draw_sensors=self.show_sensors)

        self._hud.draw(
            self._screen,
            speed    = self.car.speed,
            reward   = self._total_reward,
            total_r  = self._total_reward,
            step     = self._step,
            episode  = self._episode,
            laps     = self.track.lap_count,
            steering = self._last_action,
        )

        pygame.display.flip()
        fps = 15 if self.slow_mo else self.metadata["render_fps"]
        self._clock.tick(fps)

        if self.render_mode == "rgb_array":
            return pygame.surfarray.array3d(self._screen).transpose(1, 0, 2)

    def close(self):
        if self._screen is not None:
            pygame.quit()
            self._screen = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        sensors    = self.car.cast_sensors(self.track.wall_segments)
        front      = float(sensors[2])
        left       = float(sensors[1])
        right      = float(sensors[3])
        speed_norm = float(self.car.speed / Car.MAX_SPEED)
        angle_diff = self.track.angle_to_center(self.car.x, self.car.y, self.car.angle)
        angle_norm = float(math.tanh(angle_diff))
        return np.array([front, left, right, speed_norm, angle_norm],
                        dtype=np.float32)

    def _is_crashed(self) -> bool:
        """
        Check each car corner against the track surface pixel colour.
        Off-road pixel (not grey road colour) → crash.
        """
        for cx, cy in self.car.get_corners():
            if not self.track.is_on_road(cx, cy):
                return True
        return False

    def _init_display(self):
        """Open a real Pygame display window for human rendering."""
        if not pygame.get_init():
            pygame.init()
        self._screen = pygame.display.set_mode((self.track.W, self.track.H))
        pygame.display.set_caption("PPO Self-Driving Car")
        self._clock  = pygame.time.Clock()
        self._hud    = HUD()
        # Rebuild surface onto the display (replaces the offscreen one)
        self.track._track_surf = None
        self.track.build_surface()
