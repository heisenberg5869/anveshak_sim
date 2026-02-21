from odometry import Odometry
from perception.obstacle_detector import ObstacleDetector
from visualize import Visualizer


class Robot:
    def __init__(self):
        self.odom = Odometry()
        self.detector = ObstacleDetector()
        self.viz = Visualizer()

    def step(
        self,
        lidar_points,
        lidar_rays,
        lidar_hits,
        v,
        w,
        dt,
        show_lidar=True,
        show_odom=True
    ):
        """
        Advance the robot state by one timestep.

        lidar_points : LiDAR points in robot frame
        lidar_rays   : LiDAR rays in world frame (visualization)
        lidar_hits   : LiDAR hit points in world frame
        v            : linear velocity command
        w            : angular velocity command
        dt           : timestep
        """

        # Basic perception from LiDAR
        obstacles = self.detector.detect(lidar_points)

        # Update ground truth and odometry
        self.odom.update(v, w, dt)

        # Update visualization
        self.viz.update(
            self.odom,
            lidar_points,
            lidar_rays,
            lidar_hits,
            obstacles,
            show_lidar=show_lidar,
            show_odom=show_odom
        )

    # ----------------------------
    # Pose accessors
    # ----------------------------
    def get_ground_truth(self):
        return self.odom.gt_x, self.odom.gt_y, self.odom.gt_theta

    def get_odometry(self):
        return self.odom.x, self.odom.y, self.odom.theta
