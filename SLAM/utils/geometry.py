import math


def normalize_angle(theta):
    return math.atan2(math.sin(theta), math.cos(theta))
