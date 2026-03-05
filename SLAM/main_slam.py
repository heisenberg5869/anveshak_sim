from robot import Robot
from sensors.lidar import LidarScan
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import math
import csv

# Runtime modes & visualization toggles
MODE = "MANUAL"        # MANUAL | AUTO
SHOW_LIDAR = True
SHOW_ODOM = True
SHOW_MAP  = True       # Toggle occupancy map overlay with 'm' 

# Control commands
v = 0.0   # linear velocity [m/s]
w = 0.0   # angular velocity [rad/s]

# ---------------------------------------------------------------------------
# Occupancy Grid Parameters
# ---------------------------------------------------------------------------
GRID_RESOLUTION = 0.05          # metres per cell
GRID_WIDTH_M    = 12.0          # total map width  [m]
GRID_HEIGHT_M   = 12.0          # total map height [m]
GRID_ORIGIN_X   = -GRID_WIDTH_M  / 2.0   # world-x of cell (0,0)
GRID_ORIGIN_Y   = -GRID_HEIGHT_M / 2.0   # world-y of cell (0,0)

GRID_COLS = int(GRID_WIDTH_M  / GRID_RESOLUTION)
GRID_ROWS = int(GRID_HEIGHT_M / GRID_RESOLUTION)

# Log-odds update values  
L_OCC   =  0.85    # log-odds added when a cell is hit
L_FREE  = -0.40    # log-odds added when a cell is on the free ray
L_MIN   = -5.0     # clamp floor
L_MAX   =  5.0     # clamp ceiling

# Observation confidence as a function of range
MAX_RANGE_CONF = 4.0   # metres – range at which confidence reaches 0

# Motion-gating threshold (skip update when spinning fast)
ANGULAR_GATE = 0.5     # rad/s

# Wall preservation: if a cell already looks occupied, be conservative
# about clearing it from a new angle (only clear if scan angle is similar)
WALL_ANGLE_THRESHOLD_DEG = 30.0

# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def world_to_grid(wx, wy):
    """Convert world coords (metres) → grid indices (col, row).  Returns None if OOB."""
    col = int((wx - GRID_ORIGIN_X) / GRID_RESOLUTION)
    row = int((wy - GRID_ORIGIN_Y) / GRID_RESOLUTION)
    if 0 <= col < GRID_COLS and 0 <= row < GRID_ROWS:
        return col, row
    return None


def grid_to_world(col, row):
    """Centre of grid cell → world coordinates."""
    wx = GRID_ORIGIN_X + (col + 0.5) * GRID_RESOLUTION
    wy = GRID_ORIGIN_Y + (row + 0.5) * GRID_RESOLUTION
    return wx, wy


def bresenham(c0, r0, c1, r1):
    """Bresenham's line algorithm; yields (col, row) from start to end (inclusive)."""
    cols, rows = [], []
    dc = abs(c1 - c0); sc = 1 if c1 > c0 else -1
    dr = abs(r1 - r0); sr = 1 if r1 > r0 else -1
    err = dc - dr
    c, r = c0, r0
    while True:
        cols.append(c); rows.append(r)
        if c == c1 and r == r1:
            break
        e2 = 2 * err
        if e2 > -dr: err -= dr; c += sc
        if e2 <  dc: err += dc; r += sr
    return list(zip(cols, rows))


# ---------------------------------------------------------------------------
# Core mapping update
# ---------------------------------------------------------------------------

def update_occupancy_grid(log_odds, detect_angle,
                          lidar_ranges, lidar_points,
                          rover_x, rover_y, rover_theta,
                          rover_angular_z, max_range):
    """
    Bayesian occupancy grid update using log-odds representation.

    Parameters
    ----------
    log_odds      : 2-D np.ndarray (GRID_ROWS × GRID_COLS) of log-odds values
    detect_angle  : 2-D np.ndarray storing the last detection angle per cell
    lidar_ranges  : 1-D array of range measurements from the LiDAR
    lidar_points  : list/array of (x, y) hit points in *world* coordinates
    rover_x/y/theta : current ground-truth pose
    rover_angular_z : angular velocity for motion gating
    max_range       : sensor maximum range

    Returns
    -------
    log_odds, detect_angle  (updated in-place, also returned for clarity)
    """

    # 1. Motion gating – skip noisy updates while spinning fast
    if abs(rover_angular_z) > ANGULAR_GATE:
        return log_odds, detect_angle

    # 2. Grid cell of the rover
    rover_cell = world_to_grid(rover_x, rover_y)
    if rover_cell is None:
        return log_odds, detect_angle
    rover_col, rover_row = rover_cell

    # 3. Process each LiDAR beam
    for i, (rng, hit_world) in enumerate(zip(lidar_ranges, lidar_points)):

        # Skip invalid / max-range readings
        if rng < 0.01 or rng >= max_range * 0.99:
            continue

        hit_x, hit_y = float(hit_world[0]), float(hit_world[1])

        # Distance-based confidence: closer = more confident
        confidence = max(0.1, 1.0 - rng / MAX_RANGE_CONF)

        hit_cell = world_to_grid(hit_x, hit_y)
        if hit_cell is None:
            continue
        hit_col, hit_row = hit_cell

        # 4. Ray-cast: mark free cells along the beam
        ray_cells = bresenham(rover_col, rover_row, hit_col, hit_row)

        for (c, r) in ray_cells[:-1]:   # all cells except the endpoint
            if not (0 <= c < GRID_COLS and 0 <= r < GRID_ROWS):
                continue

            free_update = confidence * L_FREE   # negative update

            # Wall preservation: if cell looks occupied AND this ray comes
            # from a very different angle, damp the freeing update heavily.
            if log_odds[r, c] > 1.5:   # cell is probably a wall
                current_angle = math.atan2(r - rover_row, c - rover_col)
                prev_angle    = detect_angle[r, c]
                angle_diff    = abs(math.degrees(
                    math.atan2(math.sin(current_angle - prev_angle),
                               math.cos(current_angle - prev_angle))))
                if angle_diff < WALL_ANGLE_THRESHOLD_DEG:
                    free_update *= 0.1   # nearly same direction → damp strongly

            log_odds[r, c] = np.clip(log_odds[r, c] + free_update, L_MIN, L_MAX)

        # 5. Mark hit cell as occupied
        occ_update = confidence * L_OCC
        log_odds[hit_row, hit_col] = np.clip(
            log_odds[hit_row, hit_col] + occ_update, L_MIN, L_MAX)

        # Record the angle from which this cell was last detected
        detect_angle[hit_row, hit_col] = math.atan2(
            hit_row - rover_row, hit_col - rover_col)

    return log_odds, detect_angle


def log_odds_to_prob(log_odds):
    """Convert log-odds grid to probability grid [0, 1]."""
    return 1.0 - 1.0 / (1.0 + np.exp(log_odds))


# ---------------------------------------------------------------------------
# Controller (unchanged from original)
# ---------------------------------------------------------------------------

class Controller:
    def __init__(self, path_file, lookahead_dist=0.3, max_v=6.0,
                 Ld_min=0.3, Ld_max=0.6, k=0.5):
        self.path  = pd.read_csv(path_file).values
        self.Ld    = lookahead_dist
        self.v_max = max_v
        self.Ld_min = Ld_min
        self.Ld_max = Ld_max
        self.k     = k
        self.v     = 0.0
        self.w     = 0.0

    def lookahead(self, v):
        self.Ld = (self.k * v) + self.Ld_min
        self.Ld = max(min(self.Ld, self.Ld_max), self.Ld_min)

    def find_lookahead_point(self, robot_x, robot_y, current_v):
        self.lookahead(current_v)
        distances   = np.sqrt((self.path[:, 0] - robot_x)**2 +
                               (self.path[:, 1] - robot_y)**2)
        closest_idx = np.argmin(distances)

        for i in range(closest_idx, len(self.path) - 1):
            p1 = self.path[i]; p2 = self.path[i + 1]
            d = p2 - p1
            f = p1 - np.array([robot_x, robot_y])
            a = np.dot(d, d)
            b = 2 * np.dot(f, d)
            c = np.dot(f, f) - self.Ld**2
            discriminant = b**2 - 4*a*c
            if discriminant >= 0 and a > 1e-8:
                t2 = (-b + np.sqrt(discriminant)) / (2*a)
            else:
                t2 = -1.0
            if 0 <= t2 <= 1:
                return p1 + t2 * d

        return self.path[min(closest_idx + 1, len(self.path) - 1)]

    def get_control(self, robot_pose):
        x, y, theta = robot_pose
        target_pt   = self.find_lookahead_point(x, y, self.v)
        dx = target_pt[0] - x; dy = target_pt[1] - y
        local_x =  dx * np.cos(theta) + dy * np.sin(theta)
        local_y = -dx * np.sin(theta) + dy * np.cos(theta)
        kappa    = (2 * local_y) / (self.Ld**2)
        self.v   = self.v_max
        self.w   = self.v * kappa
        return self.v, self.w


# ---------------------------------------------------------------------------
# Key handler
# ---------------------------------------------------------------------------

def on_key(event):
    global v, w, MODE, SHOW_LIDAR, SHOW_ODOM, SHOW_MAP

    if event.key == 'o':
        SHOW_ODOM = not SHOW_ODOM
        print(f"Odometry visualization: {'ON' if SHOW_ODOM else 'OFF'}")
        return
    if event.key == 'l':
        SHOW_LIDAR = not SHOW_LIDAR
        print(f"LiDAR visualization: {'ON' if SHOW_LIDAR else 'OFF'}")
        return
    if event.key == 'g':
        SHOW_MAP = not SHOW_MAP
        print(f"Occupancy map: {'ON' if SHOW_MAP else 'OFF'}")
        return
    if event.key == 'm':
        MODE = "MANUAL"; v = 0.0; w = 0.0
        print("Switched to MANUAL mode"); return
    if event.key == 'a':
        MODE = "AUTO"
        print("Switched to AUTO mode"); return
    if MODE != "MANUAL":
        return
    if event.key == 'up':     v += 1.5
    elif event.key == 'down': v -= 1.5
    elif event.key == 'left': w += 2.0
    elif event.key == 'right':w -= 2.0
    elif event.key == ' ':    v = 0.0; w = 0.0
    v = max(min(v, 6.0), -6.0)
    w = max(min(w, 6.0), -6.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    lidar = LidarScan(max_range=4.0)
    robot = Robot()

    # Initialise occupancy grid (log-odds, all zeros = 0.5 probability)
    log_odds     = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
    detect_angle = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)

    plt.close('all')
    fig, (ax_sim, ax_map) = plt.subplots(1, 2, num=2, figsize=(14, 7))
    fig.canvas.manager.set_window_title("Autonomy Debug View")
    fig.canvas.mpl_connect("key_press_event", on_key)

    # Set up the map axes once
    ax_map.set_title("Occupancy Grid Map")
    ax_map.set_xlabel("X [m]"); ax_map.set_ylabel("Y [m]")
    extent = [GRID_ORIGIN_X, GRID_ORIGIN_X + GRID_WIDTH_M,
              GRID_ORIGIN_Y, GRID_ORIGIN_Y + GRID_HEIGHT_M]
    map_img = ax_map.imshow(
        np.full((GRID_ROWS, GRID_COLS), 0.5),
        cmap='gray', vmin=0.0, vmax=1.0,
        origin='lower', extent=extent, interpolation='nearest')
    rover_dot_map, = ax_map.plot([], [], 'ro', markersize=5, label='Rover')
    ax_map.legend(loc='upper right', fontsize=7)

    # Map refresh counter – update visualisation every N steps (cheap)
    MAP_REFRESH_EVERY = 10
    step_count = 0

    # Estimate angular velocity from pose changes
    prev_theta  = None
    angular_z   = 0.0

    dt = 0.01

    plt.show(block=False)

    # -----------------------------------------------------------------------
    # Main simulation loop
    # -----------------------------------------------------------------------
    while plt.fignum_exists(fig.number):

        # Ground-truth pose
        real_x, real_y, real_theta = robot.get_ground_truth()
        # Odometry estimate
        ideal_x, ideal_y, ideal_theta = robot.get_odometry()
        # LiDAR scan
        lidar_ranges, lidar_points, lidar_rays, lidar_hits = lidar.get_scan(
            (real_x, real_y, real_theta))

        # Estimate angular velocity from successive theta readings
        if prev_theta is not None:
            d_theta   = real_theta - prev_theta
            # Wrap to [-pi, pi]
            d_theta   = math.atan2(math.sin(d_theta), math.cos(d_theta))
            angular_z = d_theta / dt
        prev_theta = real_theta

        # -------------------------------------------------------------------
        # Occupancy grid update
        # -------------------------------------------------------------------
        log_odds, detect_angle = update_occupancy_grid(
            log_odds, detect_angle,
            lidar_ranges, lidar_points,
            real_x, real_y, real_theta,
            angular_z,
            lidar.max_range)

        # -------------------------------------------------------------------
        # Autonomous control
        # -------------------------------------------------------------------
        if MODE == "AUTO":
            if 'controller' not in globals():
                controller = Controller(
                    path_file=r"C:\Users\vtsar\projects\anveshak\anveshak_sim\path.csv",
                    lookahead_dist=0.4, max_v=6.0, Ld_min=0.3, Ld_max=1.0, k=0.5)

            if 'avoid_mode' not in globals():
                avoid_mode = False

            CENTER_RAY_IDX = 18
            RAY_OFFSET     = 1
            A_MAX          = 8.0
            MIN_D_STOP     = 0.50
            MAX_D_STOP     = 4.0
            AVOID_V        = 0.5
            AVOID_W        = 2.2
            OBSTACLE_DIST  = 0.65

            current_speed = abs(v)
            d_stop = (current_speed**2) / (2*A_MAX) + MIN_D_STOP
            d_stop = min(MAX_D_STOP, d_stop)

            lidar_array   = np.array(lidar_ranges)
            forward_rays  = lidar_array[
                CENTER_RAY_IDX - RAY_OFFSET: CENTER_RAY_IDX + RAY_OFFSET + 1]
            valid_forward = forward_rays[
                (forward_rays > 0.01) & (forward_rays < lidar.max_range * 0.99)]

            obstacle_detected = (len(valid_forward) >= 1 and
                                  np.any(valid_forward < d_stop))

            if obstacle_detected:
                avoid_mode = True
            elif (avoid_mode and len(valid_forward) > 0 and
                  np.min(valid_forward) > OBSTACLE_DIST * 1.5):
                avoid_mode = False

            if not avoid_mode:
                v, w = controller.get_control((ideal_x, ideal_y, ideal_theta))
            else:
                shifted     = np.roll(lidar_array, -CENTER_RAY_IDX)
                left_clear  = np.min(shifted[8:25])
                right_clear = np.min(shifted[-25:-8])
                front_clear = np.min(shifted[0:8])

                if front_clear > OBSTACLE_DIST * 1.2:
                    v = controller.v_max * 0.8; w = 0.0
                elif abs(left_clear - right_clear) > 0.2:
                    turn_dir = np.sign(left_clear - right_clear)
                    v = AVOID_V; w = turn_dir * AVOID_W
                else:
                    v = AVOID_V * 0.5; w = 0.0

        # -------------------------------------------------------------------
        # Periodically update the occupancy map visualisation
        # -------------------------------------------------------------------
        step_count += 1
        if SHOW_MAP and step_count % MAP_REFRESH_EVERY == 0:
            prob_map = log_odds_to_prob(log_odds)
            # Flip so that high occupancy = dark (0) and free = light (1)
            map_img.set_data(1.0 - prob_map)
            rover_dot_map.set_data([real_x], [real_y])
            ax_map.set_title(
                f"Occupancy Grid  (step {step_count})  "
                f"Press 'g' to toggle")
            fig.canvas.draw_idle()

        # -------------------------------------------------------------------
        # Robot step & render (don't edit below)
        # -------------------------------------------------------------------
        robot.step(
            lidar_points, lidar_rays, lidar_hits,
            v, w, dt,
            show_lidar=SHOW_LIDAR,
            show_odom=SHOW_ODOM)

        plt.pause(dt)