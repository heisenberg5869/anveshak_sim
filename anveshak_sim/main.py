from robot import Robot
from sensors.lidar import LidarScan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import csv

# Runtime modes & visualization toggles
MODE = "MANUAL"        # MANUAL | AUTO
SHOW_LIDAR = True
SHOW_ODOM = True

# Control commands
v = 0.0   # linear velocity [m/s]
w = 0.0   # angular velocity [rad/s]

class Controller:
    def __init__(self, path_file, lookahead_dist=0.3, max_v=6.0, Ld_min=0.3, Ld_max=0.6, k=0.5):
        self.path = pd.read_csv(path_file).values  
        self.Ld = lookahead_dist
        self.v_max = max_v
        self.Ld_min = Ld_min
        self.Ld_max = Ld_max
        self.k = k
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
            
            d = p2 - p1                                              # Direction vector of segment
            f = p1 - np.array([robot_x, robot_y])                     # Vector from robot to p1
            
            a = np.dot(d, d)                                          # Segment length squared
            b = 2 * np.dot(f, d)                                      # Projection of f onto d
            c = np.dot(f, f) - self.Ld**2                             # Perpendicular distance squared
        
            discriminant = b**2 - 4*a*c
            if discriminant >= 0 and a > 1e-8:
                discriminant = np.sqrt(discriminant)
                t2 = (-b + discriminant) / (2*a)                       # Intersection factor
            else:
                t2 = -1.0
            if 0 <= t2 <= 1:
                return p1 + t2 * d
                
        return self.path[closest_idx + 1]                            # Return next closest point

    def get_control(self, robot_pose):
        x, y, theta = robot_pose
        target_pt = self.find_lookahead_point(x, y, self.v)

        dx = target_pt[0] - x
        dy = target_pt[1] - y

        local_x = dx * np.cos(theta) + dy * np.sin(theta)
        local_y = -dx * np.sin(theta) + dy * np.cos(theta)

        kappa = (2 * local_y) / (self.Ld**2)

        self.v = self.v_max
        self.w = self.v * kappa

        return self.v, self.w

def on_key(event):
    """Keyboard control & visualization toggles."""
    global v, w, MODE, SHOW_LIDAR, SHOW_ODOM

    # visualization toggles 
    if event.key == 'o':
        SHOW_ODOM = not SHOW_ODOM
        print(f"Odometry visualization: {'ON' if SHOW_ODOM else 'OFF'}")
        return

    if event.key == 'l':
        SHOW_LIDAR = not SHOW_LIDAR
        print(f"LiDAR visualization: {'ON' if SHOW_LIDAR else 'OFF'}")
        return

    # mode switching 
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

    # manual control 
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

    # Main simulation loop
    while plt.fignum_exists(fig.number):

        # ground truth pose
        real_x, real_y, real_theta = robot.get_ground_truth()
        # odometry estimate
        ideal_x, ideal_y, ideal_theta = robot.get_odometry()
        # LiDAR scan 
        lidar_ranges, lidar_points, lidar_rays, lidar_hits = lidar.get_scan((real_x, real_y, real_theta))




    # write your autonomous code here!!!!!!!!!!!!!
        if MODE == "AUTO":
            if 'controller' not in globals():
                controller = Controller(path_file=r"C:\Users\vtsar\projects\anveshak\anveshak_sim\path.csv",lookahead_dist=0.4,max_v=6.0,Ld_min=0.3,Ld_max=1.0,k=0.5)

            if 'avoid_mode' not in globals():
                avoid_mode = False

            
            CENTER_RAY_IDX    = 18          # straight ahead (adjust if your lidar indexing is different)
            RAY_OFFSET        = 1           # so we check CENTER-1, CENTER, CENTER+1
            A_MAX             = 8.0         # m/s² — maximum deceleration we assume the rover can do
            MIN_D_STOP        = 0.50        # meters — never allow closer than this, even at v≈0
            MAX_D_STOP        = 4.0         # meters — upper limit so we don't over-react at high speed
            AVOID_V           = 0.5         # slow crawl while dodging
            AVOID_W           = 2.2         # sharp but controllable turn
            OBSTACLE_DIST     = 0.65        # distance to obstacle to trigger avoidance


            # Get current commanded speed 
            current_speed = abs(v)  

            # Compute required stopping distance
            d_stop = (current_speed ** 2) / (2 * A_MAX) + MIN_D_STOP
            d_stop = min(MAX_D_STOP, d_stop)

            # Get the three forward rays
            lidar_array = np.array(lidar_ranges)
            forward_rays = lidar_array[CENTER_RAY_IDX - RAY_OFFSET : CENTER_RAY_IDX + RAY_OFFSET + 1]

            # Valid ranges only
            valid_forward = forward_rays[(forward_rays > 0.01) & (forward_rays < lidar.max_range * 0.99)]

            # Detection 
            obstacle_detected = False

            if len(valid_forward) >= 1:   # at least one valid measurement
                if np.any(valid_forward < d_stop):
                    obstacle_detected = True
            
            if obstacle_detected:
                avoid_mode = True
            elif avoid_mode and len(valid_forward) > 0 and np.min(valid_forward) > OBSTACLE_DIST * 1.5:
                avoid_mode = False

            if not avoid_mode:
                v, w = controller.get_control((ideal_x, ideal_y, ideal_theta))
            else:
                shifted = np.roll(lidar_array, -CENTER_RAY_IDX)

                left_clear = np.min(shifted[8:25])    
                right_clear = np.min(shifted[-25:-8])  
                front_clear = np.min(shifted[0:8])     

                if front_clear > OBSTACLE_DIST * 1.2:
                    v = controller.v_max * 0.8 
                    w = 0.0  
                elif abs(left_clear - right_clear) > 0.2:
                    turn_dir = np.sign(left_clear - right_clear)
                    v = AVOID_V
                    w = turn_dir * AVOID_W
                else:
                    v = AVOID_V * 0.5
                    w = 0.0
                
        # don't edit below this line (visualization & robot stepping)
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
