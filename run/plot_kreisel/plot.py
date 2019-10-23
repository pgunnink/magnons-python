import os

import matplotlib.pyplot as plt
import numpy as np

from magnons.process import Process

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))

    p = Process(dir_path)
    for i, (kvalues, E, ev, attrs) in enumerate(p.get_all()):
        plt.subplot(2, 2, i + 1)
        kabs = np.sqrt(np.sum(kvalues**2, axis=1))

        for n in range(6):
            plt.semilogx(kabs, E[:, n])

    plt.show()
