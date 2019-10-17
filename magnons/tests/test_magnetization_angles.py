from magnons.magnetization import magnetization_angle
from numpy import pi
from math import radians, degrees
if __name__ == '__main__':
    # parameters taken from Unconventional spin currents in magnetic films
    # (https://arxiv.org/abs/1904.12610)
    H = 2500
    M = 1750 / (4 * pi)
    alpha = [0, 30, 60, 80]
    # should give 0, 18, 40, 65
    for a in alpha:
        print(
            f"Alpha: {a}, phi: {degrees(magnetization_angle(radians(a), M=M, H=H))}"
        )
