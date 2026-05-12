import math
import pygame
import numpy as np

# ---------------------------------------------------------------------------
# Colour palette  — keep consistent with is_road_pixel()
# ---------------------------------------------------------------------------
GRASS_COLOR      = (34, 139, 34)
ROAD_COLOR       = (80, 80, 80)
BORDER_COLOR     = (255, 255, 255)
CENTER_COLOR     = (255, 220, 0)
CHECKPOINT_COLOR = (0, 200, 255)


def is_road_pixel(r, g, b):
    """True if pixel is the road-grey colour."""
    return 60 <= r <= 110 and 60 <= g <= 110 and 60 <= b <= 110


# ---------------------------------------------------------------------------
# Track geometry builder
# ---------------------------------------------------------------------------

def _lerp_pts(p1, p2, n=15):
    if n <= 0: return []
    return [(p1[0] + (p2[0]-p1[0])*i/n,
             p1[1] + (p2[1]-p1[1])*i/n) for i in range(n)]

def _arc_pts(cx, cy, r, start_angle, end_angle, n=15):
    return [(cx + r * math.cos(start_angle + (end_angle - start_angle) * i / n),
             cy + r * math.sin(start_angle + (end_angle - start_angle) * i / n))
            for i in range(n)]


def _build_curved_track(W=800, H=600, margin=60, road_w=90, r_out=140):
    """
    Curved track (rounded rectangle) to avoid 90-degree corners.
    Returns outer_pts, inner_pts, center_pts, checkpoints, spawn(x,y,angle).
    """
    r_in = r_out - road_w
    r_cen = r_out - road_w / 2

    ol, ot = margin, margin
    or_, ob = W - margin, H - margin

    # Centers of the four corners
    tl_c = (ol + r_out, ot + r_out)
    tr_c = (or_ - r_out, ot + r_out)
    br_c = (or_ - r_out, ob - r_out)
    bl_c = (ol + r_out, ob - r_out)

    def make_loop(r, steps_arc=15, steps_straight=15):
        pts = []
        # Top straight (left to right)
        pts += _lerp_pts((tl_c[0], tl_c[1] - r), (tr_c[0], tr_c[1] - r), steps_straight)
        # Top-Right corner
        pts += _arc_pts(tr_c[0], tr_c[1], r, -math.pi/2, 0, steps_arc)
        # Right straight (top to bottom)
        pts += _lerp_pts((tr_c[0] + r, tr_c[1]), (br_c[0] + r, br_c[1]), steps_straight)
        # Bottom-Right corner
        pts += _arc_pts(br_c[0], br_c[1], r, 0, math.pi/2, steps_arc)
        # Bottom straight (right to left)
        pts += _lerp_pts((br_c[0], br_c[1] + r), (bl_c[0], bl_c[1] + r), steps_straight)
        # Bottom-Left corner
        pts += _arc_pts(bl_c[0], bl_c[1], r, math.pi/2, math.pi, steps_arc)
        # Left straight (bottom to top)
        pts += _lerp_pts((bl_c[0] - r, bl_c[1]), (tl_c[0] - r, tl_c[1]), steps_straight)
        # Top-Left corner
        pts += _arc_pts(tl_c[0], tl_c[1], r, math.pi, math.pi*1.5, steps_arc)
        return pts

    outer = make_loop(r_out, 15, 2)
    inner = make_loop(r_in, 15, 2)
    center = make_loop(r_cen, 20, 20)

    n_cp = 12
    step = max(1, len(center) // n_cp)
    checkpoints = [center[i * step] for i in range(n_cp)]

    # Spawn: middle of top straight, pointing RIGHT
    spawn_x = (tl_c[0] + tr_c[0]) / 2
    spawn_y = float(tl_c[1] - r_cen)
    spawn_angle = 0.0

    return outer, inner, center, checkpoints, (spawn_x, spawn_y, spawn_angle)


# ---------------------------------------------------------------------------
# Wall-segment helper (for raycasting)
# ---------------------------------------------------------------------------

def _poly_to_segments(pts):
    return [(pts[i], pts[(i+1) % len(pts)]) for i in range(len(pts))]


# ---------------------------------------------------------------------------
# Track class
# ---------------------------------------------------------------------------

class Track:
    CHECKPOINT_DIST = 40

    def __init__(self, width: int = 800, height: int = 600):
        self.W = width
        self.H = height

        (self.outer_pts,
         self.inner_pts,
         self.center_pts,
         self._cp_positions,
         self._spawn) = _build_curved_track(width, height)

        self.wall_segments = (
            _poly_to_segments(self.outer_pts) +
            _poly_to_segments(self.inner_pts)
        )

        self.num_checkpoints = len(self._cp_positions)
        self._track_surf = None
        self.reset_checkpoints()

    # ------------------------------------------------------------------
    # Spawn
    # ------------------------------------------------------------------
    def get_spawn(self, randomise: bool = False):
        sx, sy, sa = self._spawn
        if randomise:
            idx = np.random.randint(0, len(self.center_pts))
            p1  = self.center_pts[idx]
            p2  = self.center_pts[(idx + 1) % len(self.center_pts)]
            sx, sy = p1
            sa = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
            sa += np.random.uniform(-0.15, 0.15)
        return sx, sy, sa

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------
    def reset_checkpoints(self):
        self._cp_hit    = [False] * self.num_checkpoints
        self._next_cp   = 0
        self._lap_count = 0

    def check_checkpoint(self, x, y) -> bool:
        cx, cy = self._cp_positions[self._next_cp]
        if math.hypot(x - cx, y - cy) < self.CHECKPOINT_DIST:
            self._cp_hit[self._next_cp] = True
            self._next_cp += 1
            if self._next_cp >= self.num_checkpoints:
                self._next_cp   = 0
                self._lap_count += 1
                self._cp_hit    = [False] * self.num_checkpoints
            return True
        return False

    @property
    def lap_count(self):
        return self._lap_count

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def angle_to_center(self, x, y, car_angle) -> float:
        min_d, best_i = float("inf"), 0
        for i, (cx, cy) in enumerate(self.center_pts):
            d = math.hypot(x - cx, y - cy)
            if d < min_d:
                min_d, best_i = d, i
        j  = (best_i + 1) % len(self.center_pts)
        tx = self.center_pts[j][0] - self.center_pts[best_i][0]
        ty = self.center_pts[j][1] - self.center_pts[best_i][1]
        diff = math.atan2(ty, tx) - car_angle
        while diff >  math.pi: diff -= 2 * math.pi
        while diff < -math.pi: diff += 2 * math.pi
        return diff

    def dist_to_center(self, x, y) -> float:
        return min(math.hypot(x - cx, y - cy) for cx, cy in self.center_pts)

    # ------------------------------------------------------------------
    # Pixel-based on-road check
    # ------------------------------------------------------------------
    def is_on_road(self, x: float, y: float) -> bool:
        """Returns True if point (x,y) sits on a road-coloured pixel."""
        if self._track_surf is None:
            return True                     # surface not built yet — be lenient
        ix, iy = int(round(x)), int(round(y))
        if ix < 0 or iy < 0 or ix >= self.W or iy >= self.H:
            return False
        col = self._track_surf.get_at((ix, iy))
        return is_road_pixel(col[0], col[1], col[2])

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def build_surface(self) -> pygame.Surface:
        surf = pygame.Surface((self.W, self.H))
        surf.fill(GRASS_COLOR)

        pygame.draw.polygon(surf, ROAD_COLOR,
                            [(int(x), int(y)) for x, y in self.outer_pts])
        pygame.draw.polygon(surf, GRASS_COLOR,
                            [(int(x), int(y)) for x, y in self.inner_pts])

        pygame.draw.lines(surf, BORDER_COLOR, True,
                          [(int(x), int(y)) for x, y in self.outer_pts], 4)
        pygame.draw.lines(surf, BORDER_COLOR, True,
                          [(int(x), int(y)) for x, y in self.inner_pts], 4)

        n = len(self.center_pts)
        for i in range(0, n, 2):
            j = (i + 1) % n
            pygame.draw.line(surf, CENTER_COLOR,
                             (int(self.center_pts[i][0]), int(self.center_pts[i][1])),
                             (int(self.center_pts[j][0]), int(self.center_pts[j][1])), 1)

        self._track_surf = surf
        return surf

    def draw(self, surface: pygame.Surface):
        if self._track_surf is None:
            self.build_surface()
        surface.blit(self._track_surf, (0, 0))
        self._draw_checkpoints(surface)

    def _draw_checkpoints(self, surface: pygame.Surface):
        for i, (cx, cy) in enumerate(self._cp_positions):
            color = (0, 255, 180) if i == self._next_cp else (
                    (80, 80, 80) if self._cp_hit[i] else CHECKPOINT_COLOR)
            pygame.draw.circle(surface, color, (int(cx), int(cy)),
                               self.CHECKPOINT_DIST, 2)