import math
import random
from world.obstacles import get_world_obstacles


class LidarScan:
    def __init__(self, max_range=4.0, angle_step_deg=10):
        self.max_range = max_range
        self.angle_step = angle_step_deg
        self.obstacles = get_world_obstacles()

        # fixed angle list (deterministic ordering)
        self.angles_deg = list(range(0, 360, self.angle_step))

    def get_scan(self, robot_pose):
        """
        Simulate a 2D LiDAR scan.

        Returns:
            lidar_ranges : list[float]  (length = N beams)
            lidar_points : list[(x,y)]  robot frame (derived)
            lidar_rays   : world frame (viz only)
            lidar_hits   : world frame (viz only)
        """
        rx, ry, rtheta = robot_pose

        lidar_ranges = []
        lidar_points = []
        lidar_rays = []
        lidar_hits = []

        for angle_deg in self.angles_deg:
            ray_angle = math.radians(angle_deg) + rtheta

            step = 0.05
            d_prev = 0.0
            d = 0.0
            hit = False

            # coarse ray march (real geometry)
            while d < self.max_range:
                px = rx + d * math.cos(ray_angle)
                py = ry + d * math.sin(ray_angle)

                for obs in self.obstacles:
                    if obs.contains(px, py):
                        hit = True
                        break

                if hit:
                    break

                d_prev = d
                d += step

            # refine hit location
            if hit:
                lo, hi = d_prev, d
                for _ in range(10):
                    mid = 0.5 * (lo + hi)
                    mx = rx + mid * math.cos(ray_angle)
                    my = ry + mid * math.sin(ray_angle)

                    if any(obs.contains(mx, my) for obs in self.obstacles):
                        hi = mid
                    else:
                        lo = mid

                dist = lo
            else:
                dist = self.max_range

            # add measurement noise
            dist += random.gauss(0.0, 0.01)
            dist = max(0.0, min(dist, self.max_range))

            # --- store REAL lidar data ---
            lidar_ranges.append(dist)

            # robot-frame point (helper)
            lx = dist * math.cos(ray_angle - rtheta)
            ly = dist * math.sin(ray_angle - rtheta)
            lidar_points.append((lx, ly))

            # visualization helpers
            x_end = rx + dist * math.cos(ray_angle)
            y_end = ry + dist * math.sin(ray_angle)
            lidar_rays.append(((rx, ry), (x_end, y_end)))

            if hit:
                lidar_hits.append((x_end, y_end))

        return lidar_ranges, lidar_points, lidar_rays, lidar_hits
