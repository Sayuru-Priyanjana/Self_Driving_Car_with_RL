"""
car.py — Car physics and sensor system for the self-driving mini car.
"""

import math
import numpy as np
import pygame


class Car:
    """2D top-down car with simple physics and raycast sensors."""

    # --- tuneable constants ---
    MAX_SPEED     = 4.5
    ACCELERATION  = 0.18
    FRICTION      = 0.92
    TURN_RATE     = 0.06   # radians per step at full steering

    # sensor geometry
    NUM_SENSORS   = 5
    SENSOR_LEN    = 150
    SENSOR_ANGLES = [-math.pi / 2, -math.pi / 4, 0, math.pi / 4, math.pi / 2]

    # visual size
    CAR_W = 14
    CAR_H = 22

    def __init__(self, x: float, y: float, angle: float = 0.0):
        self.x     = x
        self.y     = y
        self.angle = angle   # radians; 0 = pointing right
        self.speed = 0.0
        self.alive = True

        # sensor readings (normalised 0-1)
        self.sensor_dists = [1.0] * self.NUM_SENSORS

    # ------------------------------------------------------------------
    # physics
    # ------------------------------------------------------------------
    def step(self, steering: float, throttle: float = 1.0):
        """
        Advance physics by one simulation step.
        steering : float in [-1, 1]
        throttle : float in [ 0, 1]  (fixed to 1 for single-action mode)
        """
        # accelerate then friction
        self.speed += throttle * self.ACCELERATION
        self.speed  = min(self.speed, self.MAX_SPEED)
        self.speed *= self.FRICTION

        # steer
        self.angle += steering * self.TURN_RATE * (self.speed / self.MAX_SPEED + 0.1)

        # translate
        self.x += self.speed * math.cos(self.angle)
        self.y += self.speed * math.sin(self.angle)

    # ------------------------------------------------------------------
    # sensors — simple raycasts against a list of wall segments
    # ------------------------------------------------------------------
    def cast_sensors(self, wall_segments: list) -> np.ndarray:
        """
        Cast NUM_SENSORS rays from car centre, return normalised distances [0,1].
        wall_segments : list of ((x1,y1),(x2,y2)) tuples.
        """
        readings = []
        for rel_angle in self.SENSOR_ANGLES:
            world_angle = self.angle + rel_angle
            dist = self._cast_single(world_angle, wall_segments)
            readings.append(dist / self.SENSOR_LEN)
        self.sensor_dists = readings
        return np.array(readings, dtype=np.float32)

    def _cast_single(self, angle: float, segments: list) -> float:
        """Return distance along `angle` to nearest wall (up to SENSOR_LEN)."""
        rx = math.cos(angle)
        ry = math.sin(angle)
        min_t = self.SENSOR_LEN

        for (x1, y1), (x2, y2) in segments:
            # segment parametric: P = (x1,y1) + u*(x2-x1,y2-y1)
            # ray:                 Q = (cx,cy) + t*(rx,ry)
            dx = x2 - x1
            dy = y2 - y1
            denom = rx * dy - ry * dx
            if abs(denom) < 1e-6:
                continue
            t = ((x1 - self.x) * dy - (y1 - self.y) * dx) / denom
            u = ((x1 - self.x) * ry - (y1 - self.y) * rx) / denom
            if 0 < t < min_t and 0.0 <= u <= 1.0:
                min_t = t

        return min_t

    # ------------------------------------------------------------------
    # collision
    # ------------------------------------------------------------------
    def get_corners(self) -> list:
        """Return the four corners of the car rectangle in world space."""
        w, h = self.CAR_W / 2, self.CAR_H / 2
        corners_local = [(-h, -w), (h, -w), (h, w), (-h, w)]
        cos_a, sin_a = math.cos(self.angle), math.sin(self.angle)
        corners = []
        for lx, ly in corners_local:
            wx = self.x + cos_a * lx - sin_a * ly
            wy = self.y + sin_a * lx + cos_a * ly
            corners.append((wx, wy))
        return corners

    def check_collision(self, track_surface: pygame.Surface) -> bool:
        """
        Sample pixels at the car's corners.  If any corner lands on the
        grass colour (off-road) we declare a collision.
        Returns True when the car has crashed.
        """
        if track_surface is None:
            return False
        w, h = track_surface.get_size()
        for cx, cy in self.get_corners():
            ix, iy = int(cx), int(cy)
            if ix < 0 or iy < 0 or ix >= w or iy >= h:
                return True
            r, g, b, *_ = track_surface.get_at((ix, iy))
            # grass colour defined in track.py: (34, 139, 34)
            if r < 80 and g > 80 and b < 80:   # greenish → off-road
                return True
        return False

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
    def draw(self, surface: pygame.Surface, draw_sensors: bool = True):
        """Draw the car and, optionally, its sensor rays."""
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        w, h  = self.CAR_W / 2, self.CAR_H / 2

        # body polygon
        corners_local = [(-h, -w), (h, -w), (h, w), (-h, w)]
        pts = []
        for lx, ly in corners_local:
            wx = self.x + cos_a * lx - sin_a * ly
            wy = self.y + sin_a * lx + cos_a * ly
            pts.append((wx, wy))

        pygame.draw.polygon(surface, (220, 50, 50), pts)
        pygame.draw.polygon(surface, (255, 200, 200), pts, 2)

        # direction indicator (front)
        fx = self.x + cos_a * h * 1.3
        fy = self.y + sin_a * h * 1.3
        pygame.draw.circle(surface, (255, 255, 100), (int(fx), int(fy)), 4)

        if draw_sensors:
            for i, rel_angle in enumerate(self.SENSOR_ANGLES):
                world_angle = self.angle + rel_angle
                dist = self.sensor_dists[i] * self.SENSOR_LEN
                ex = self.x + math.cos(world_angle) * dist
                ey = self.y + math.sin(world_angle) * dist
                # colour shifts red→green with distance
                ratio = self.sensor_dists[i]
                colour = (int(255 * (1 - ratio)), int(255 * ratio), 80)
                pygame.draw.line(surface, colour,
                                 (int(self.x), int(self.y)),
                                 (int(ex), int(ey)), 1)
                pygame.draw.circle(surface, colour, (int(ex), int(ey)), 3)
