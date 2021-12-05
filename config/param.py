from enum import Enum


class Plane(Enum):
    AXIAL = 1
    SAGITTAL = 0
    CORONAL = 2


TEST_RATIO = 0.2

GAMMA = 0.1
LAMBDA = 6 * 10**(-4)

PLOTTING_RATE = 0.001

SLICE_HEIGHT = 288
SLICE_WIDTH = 288
