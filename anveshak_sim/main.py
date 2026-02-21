from robot import Robot
from sensors.lidar import LidarScan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import csv

# -----------------------------------------------------
# Runtime modes & visualization toggles
# -----------------------------------------------------
MODE = "MANUAL"        # MANUAL | AUTO
SHOW_LIDAR = True
SHOW_ODOM = True


# -----------------------------------------------------
# Control commands (shared state)
# -----------------------------------------------------
v = 0.0   # linear velocity [m/s]
w = 0.0   # angular velocity [rad/s]

class Controller:
    def __init__(self, path_file, lookahead_dist=0.3, max_v=6.0, Ld_min=0.3, Ld_max=1.0, k=0.5, Kp=1.6, Ki=0.1, Kd=0.4, dt=0.01):
        # Load (x, y) coordinates from CSV
        self.path = pd.read_csv(path_file).values  
        self.Ld = lookahead_dist
        self.v_max = max_v
        self.Ld_min = Ld_min
        self.Ld_max = Ld_max
        self.k = k
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0
        self.previous_error = 0.0
        self.derivative = 0.0
        self.dt = dt
        self.v = 0.0
        self.w = 0.0

    def lookahead(self, v):
        self.Ld = (self.k * v) + self.Ld_min
        self.Ld = max(min(self.Ld, self.Ld_max), self.Ld_min)

    def find_lookahead_point(self, robot_x, robot_y, current_v):
        self.lookahead(current_v)
        # 1. Find the point on the path closest to the robot
        distances = np.sqrt((self.path[:,0] - robot_x)**2 + (self.path[:,1] - robot_y)**2)
        closest_idx = np.argmin(distances)
    
        # 2. Search forward from the closest index for the lookahead point
        for i in range(closest_idx, len(self.path) - 1):
            p1 = self.path[i]
            p2 = self.path[i+1]
            
            d = p2 - p1 # Direction vector of segment
            f = p1 - np.array([robot_x, robot_y]) # Vector from robot to p1
            
            a = np.dot(d, d)
            b = 2 * np.dot(f, d)
            c = np.dot(f, f) - self.Ld**2
        
            discriminant = b**2 - 4*a*c
            if discriminant >= 0 and a > 10e-8:
                discriminant = np.sqrt(discriminant)
                t2 = (-b + discriminant) / (2*a) # Intersection factor
            else:
                t2 = -1.0
                if 0 <= t2 <= 1:
                    return p1 + t2 * d
                
        return self.path[closest_idx + 1] # Return next closest point

    def get_control(self, robot_pose):
        x, y, theta = robot_pose
        target_pt = self.find_lookahead_point(x, y, self.v)
    
        # Transform target to robot's local frame
        dx = target_pt[0] - x
        dy = target_pt[1] - y
    
        # Local target coordinates (Rotation matrix)
        local_x = dx * np.cos(theta) + dy * np.sin(theta)
        local_y = -dx * np.sin(theta) + dy * np.cos(theta)
    
        # Calculate Curvature (kappa = 2y / Ld^2)
        # We use local_y because it represents the lateral error
        kappa = (2 * local_y) / (self.Ld**2)

        #Calculate Error
        error = local_y

        #Calculate Derivative
        self.derivative = (error - self.previous_error) / self.dt

        #Calculate Integral
        self.integral += error * self.dt
        self.previous_error = error

        w_pid = self.Kp * error + self.Ki * self.integral + self.Kd * self.derivative
        self.v = self.v_max
        self.w = w_pid + self.v * kappa  # angular velocity
    
        return self.v, self.w

def on_key(event):
    """Keyboard control & visualization toggles."""
    global v, w, MODE, SHOW_LIDAR, SHOW_ODOM

    # --- visualization toggles ---
    if event.key == 'o':
        SHOW_ODOM = not SHOW_ODOM
        print(f"Odometry visualization: {'ON' if SHOW_ODOM else 'OFF'}")
        return

    if event.key == 'l':
        SHOW_LIDAR = not SHOW_LIDAR
        print(f"LiDAR visualization: {'ON' if SHOW_LIDAR else 'OFF'}")
        return

    # --- mode switching ---
    if event.key == 'm':
        MODE = "MANUAL"
        v = 0.0
        w = 0.0
        print("Switched to MANUAL mode")
        return

    if event.key == 'a':
        MODE = "AUTO"
        print("Switched to AUTO mode")
        return

    # --- manual control ---
    if MODE != "MANUAL":
        return

    if event.key == 'up':
        v += 1.5
    elif event.key == 'down':
        v -= 1.5
    elif event.key == 'left':
        w += 2.0
    elif event.key == 'right':
        w -= 2.0
    elif event.key == ' ':
        v = 0.0
        w = 0.0

    # clamp commands
    v = max(min(v, 6.0), -6.0)
    w = max(min(w, 6.0), -6.0)


if __name__ == "__main__":

    lidar = LidarScan(max_range=4.0)
    robot = Robot()

    plt.close('all')
    fig = plt.figure(num=2)
    fig.canvas.manager.set_window_title("Autonomy Debug View")
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show(block=False)

    dt = 0.01 

    # -------------------------------------------------
    # Main simulation loop
    # -------------------------------------------------
    while plt.fignum_exists(fig.number):

        # ground truth pose
        real_x, real_y, real_theta = robot.get_ground_truth()
        # odometry estimate
        ideal_x, ideal_y, ideal_theta = robot.get_odometry()
        # LiDAR scan 
        lidar_ranges, lidar_points, lidar_rays, lidar_hits = lidar.get_scan((real_x, real_y, real_theta))




    # ---------------------------------------------
    # write your autonomous code here!!!!!!!!!!!!!
    # ---------------------------------------------
        if MODE == "AUTO":
            if 'controller' not in globals():
                controller = Controller(path_file=r"C:\Users\vtsar\projects\anveshak\anveshak_sim\path.csv",lookahead_dist=0.4,max_v=6.0,Ld_min=0.3,Ld_max=1.0,k=0.5,Kp=1.6,Ki=0.1,Kd=0.4,dt=dt)

            if 'avoid_mode' not in globals():
                avoid_mode = False

            
            CENTER_RAY_IDX    = 18          # straight ahead (adjust if your lidar indexing is different)
            RAY_OFFSET        = 1           # so we check CENTER-1, CENTER, CENTER+1
            A_MAX             = 8.0         # m/s² — maximum deceleration we assume the robot can do
            MIN_D_STOP        = 0.20        # meters — never allow closer than this, even at v≈0
            MAX_D_STOP        = 4.0         # meters — upper limit so we don't over-react at high speed
            AVOID_V           = 0.5        # slow crawl while dodging
            AVOID_W           = 2.2         # sharp but controllable turn
            OBSTACLE_DIST = 0.65            # distance to obstacle to trigger avoidance


            # Get current commanded speed (the one we're about to send)
            # We use abs(v) because direction doesn't matter for stopping distance
            current_speed = abs(v)   # ← v is still the previous loop's value or 0 at start

            # Compute required stopping distance
            d_stop = (current_speed ** 2) / (2 * A_MAX)

            # Apply floor and ceiling
            d_stop = max(MIN_D_STOP, min(MAX_D_STOP, d_stop))

            # Get the three forward rays
            lidar_array = np.array(lidar_ranges)
            forward_rays = lidar_array[CENTER_RAY_IDX - RAY_OFFSET : CENTER_RAY_IDX + RAY_OFFSET + 1]

            # Valid ranges only (ignore max_range readings or NaN/infs if any)
            valid_forward = forward_rays[(forward_rays > 0.01) & (forward_rays < lidar.max_range * 0.99)]

            # ─── Detection ──────────────────────────────────
            obstacle_detected = False

            if len(valid_forward) >= 1:   # at least one valid measurement
                if np.any(valid_forward < d_stop):
                    obstacle_detected = True
            
            # ─── State machine ──────────────────────────────────────────────────────
            if obstacle_detected:
                avoid_mode = True
            elif not obstacle_detected:
                avoid_mode = False

            # ─── Control logic ──────────────────────────────────────────────────────
            if not avoid_mode:
                # === NORMAL PATH FOLLOWING ===
                v, w = controller.get_control((ideal_x, ideal_y, ideal_theta))
            else:
                # === REACTIVE AVOIDANCE ===
                # Clever trick: roll the array so that ray 0 = robot's current front
                shifted = np.roll(lidar_array, -CENTER_RAY_IDX)

                # Left half (positive rotation = CCW = left turn) and right half
                half = len(shifted) // 2
                left_sector  = shifted[2:half]          # left side
                right_sector = shifted[half+2:]         # right side

                # "Clearance" = how far the closest obstacle is on that side
                left_clearance  = np.min(left_sector)  if len(left_sector)  > 0 else 0.0
                right_clearance = np.min(right_sector) if len(right_sector) > 0 else 0.0

                # Choose the obviously better side
                if left_clearance > right_clearance + 0.15:      # small buffer
                    turn_dir = 1.0                               # turn left
                else:
                    turn_dir = -1.0                              # turn right

                v = AVOID_V
                w = turn_dir * AVOID_W

            
        # ---------------------------------------------
        # don't edit below this line (visualization & robot stepping)
        # ---------------------------------------------
        robot.step(
            lidar_points,
            lidar_rays,
            lidar_hits,
            v,
            w,
            dt,
            show_lidar=SHOW_LIDAR,
            show_odom=SHOW_ODOM
        )

        plt.pause(dt)
