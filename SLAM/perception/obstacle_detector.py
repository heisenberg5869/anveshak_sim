from utils.config import STOP_DISTANCE, ROBOT_RADIUS


class ObstacleDetector:
    def detect(self, lidar_points):
        """
        Detect obstacles in front of the robot using LiDAR points
        (points are in robot frame)
        """
        obstacles = []

        for x, y in lidar_points:
            # Only consider points in front
            if x <= 0:
                continue

            # Check if obstacle lies within robot width corridor
            if abs(y) <= ROBOT_RADIUS:
                if x <= STOP_DISTANCE:
                    obstacles.append((x, y))

        return obstacles
