from magnons.spin import get_spincurrent_total
from magnons.process import Process
import os
from magnons.yig import J, S
import numpy as np
import matplotlib.pyplot as plt
from magnons.plot import plot_totalspincurrent
if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))

    p = Process(dir_path)
    plot_totalspincurrent(p)
