import math
from utils.geometry import normalize_angle


class Odometry:
    def __init__(self):
        # Ground truth pose
        self.gt_x = 5.0
        self.gt_y = 2.5
        self.gt_theta = 0.0
        self.conversion_factor = 1.0 # Removed the 1.3 conversion_factor scaling

        # Estimated pose (odometry)
        self.x = 5.0
        self.y = 2.5
        self.theta = 0.0  # Removed the initial bias pi/6

    def update(self, v, omega, dt):
        """
        Update ground truth and odometry states
        using commanded linear and angular velocity.
        """

        #--------------------------------
        # Ground truth motion integration
        #--------------------------------
        self.gt_x += v * math.cos(self.gt_theta) * dt
        self.gt_y += v * math.sin(self.gt_theta) * dt
        self.gt_theta += omega * dt
        self.gt_theta = normalize_angle(self.gt_theta)

        #--------------------------------
        # Odometry motion integration
        #--------------------------------
        # Interchanged cos and sin
        self.x += v * math.cos(self.theta) * dt # Removed the +0.1 noise offset
        self.y += v * math.sin(self.theta) * dt
        self.theta += omega * dt *self.conversion_factor
        self.theta = normalize_angle(self.theta) # Added normalization to keep theta in [-pi, pi]



 

  