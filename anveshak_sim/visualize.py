import math
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from world.obstacles import get_world_obstacles


ARENA_SIZE = 20.0
LIDAR_RANGE = 4.0


class Visualizer:
    def __init__(self):
        self.real_xs = []
        self.real_ys = []
        self.idea_xs = []
        self.idea_ys = []

    def update(
        self,
        odom,
        lidar_points,
        lidar_rays,
        lidar_hits,
        obstacles,
        show_lidar=True,
        show_odom=True
    ):
        # Store trajectory history
        self.real_xs.append(odom.gt_x)
        self.real_ys.append(odom.gt_y)
        self.idea_xs.append(odom.x)
        self.idea_ys.append(odom.y)

        plt.clf()
        ax = plt.gca()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0, ARENA_SIZE)
        ax.set_ylim(0, ARENA_SIZE)
        ax.set_title("Autonomy Debug View")

        # --------------------------------
        # World obstacles
        # --------------------------------
        for obs in get_world_obstacles():
            poly = Polygon(
                obs.corners(),
                closed=True,
                facecolor="lightgray",
                edgecolor="black",
                linewidth=2.0,
                hatch="///",
                alpha=0.9,
                zorder=2
            )
            ax.add_patch(poly)

        # --------------------------------
        # LiDAR visualization
        # --------------------------------
        if show_lidar:
            # LiDAR range circle
            circle = Circle(
                (odom.gt_x, odom.gt_y),
                LIDAR_RANGE,
                edgecolor="gray",
                facecolor="none",
                linewidth=1.0,
                alpha=0.4,
                zorder=1
            )
            ax.add_patch(circle)

            # LiDAR rays
            for (x0, y0), (x1, y1) in lidar_rays:
                ax.plot(
                    [x0, x1],
                    [y0, y1],
                    color="gray",
                    linewidth=0.6,
                    alpha=0.35,
                    zorder=1
                )

            # LiDAR hit points
            if lidar_hits:
                hx, hy = zip(*lidar_hits)
                ax.scatter(
                    hx,
                    hy,
                    s=3,
                    color="gray",
                    alpha=0.9,
                    zorder=3
                )

        # --------------------------------
        # Odometry estimate (idea)
        # --------------------------------
        if show_odom:
            ax.plot(
                self.idea_xs,
                self.idea_ys,
                linestyle="--",
                color="red",
                linewidth=1.5,
                alpha=0.45,
                label="Odometry",
                zorder=3
            )
            ax.scatter(
                [odom.x],
                [odom.y],
                color="red",
                s=40,
                alpha=0.45,
                zorder=6
            )

        # --------------------------------
        # Ground truth
        # --------------------------------
        ax.plot(
            self.real_xs,
            self.real_ys,
            color="green",
            linewidth=3.0,
            label="Ground truth",
            zorder=4
        )
        ax.scatter(
            [odom.gt_x],
            [odom.gt_y],
            color="green",
            s=80,
            zorder=4
        )

        # Heading arrow
        arrow_len = 0.8
        arrow_head = 0.15
        ax.arrow(
            odom.gt_x,
            odom.gt_y,
            arrow_len * math.cos(odom.gt_theta),
            arrow_len * math.sin(odom.gt_theta),
            head_width=arrow_head,
            head_length=arrow_head,
            linewidth=2.5,
            color="green",
            zorder=5
        )

        ax.legend(loc="upper left")
        plt.pause(0.001)
