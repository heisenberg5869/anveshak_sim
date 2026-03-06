class Rectangle:
    def __init__(self, x, y, w, h):
        """
        Axis-aligned rectangle
        (x, y) = bottom-left corner
        """
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def contains(self, px, py):
        return (
            self.x <= px <= self.x + self.w and
            self.y <= py <= self.y + self.h
        )

    def corners(self):
        return [
            (self.x, self.y),
            (self.x + self.w, self.y),
            (self.x + self.w, self.y + self.h),
            (self.x, self.y + self.h),
        ]


# -----------------------------
# WORLD DEFINITION
# -----------------------------
ARENA_SIZE = 20.0
WALL_THICKNESS = 0.3


def get_world_obstacles():
    obstacles = []

    # -----------------------------
    # Outer walls (arena boundary)
    # -----------------------------
    # Bottom
    obstacles.append(Rectangle(0, 0, ARENA_SIZE, WALL_THICKNESS))
    # Top
    obstacles.append(Rectangle(0, ARENA_SIZE - WALL_THICKNESS,
                               ARENA_SIZE, WALL_THICKNESS))
    # Left
    obstacles.append(Rectangle(0, 0, WALL_THICKNESS, ARENA_SIZE))
    # Right
    obstacles.append(Rectangle(ARENA_SIZE - WALL_THICKNESS, 0,
                               WALL_THICKNESS, ARENA_SIZE))

    # -----------------------------
    # Internal obstacles
    # -----------------------------
    obstacles.extend([
        Rectangle(4, 4, 2, 6),
        Rectangle(8, 2, 3, 2),
        Rectangle(12, 5, 2, 8),
        Rectangle(5, 12, 6, 2),
        Rectangle(14, 14, 3, 3),
        Rectangle(9, 9, 2, 2),
    ])

    return obstacles
