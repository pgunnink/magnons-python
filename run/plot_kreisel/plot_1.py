import os

import matplotlib.pyplot as plt
import numpy as np

from magnons.process import Process

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))

    p = Process(dir_path)
    n = 3
    for i, (kvalues, E, ev, attrs) in enumerate(p.get_all()):
        if i == n:
            kabs = np.sqrt(np.sum(kvalues**2, axis=1))

            for n in range(6):
                plt.semilogx(kabs, E[:, n], 'black')
            plt.ylim([np.min(E) * .95, 5.5])
            plt.xlabel(r'$|k|$')
            plt.ylabel(r'E(GHz)')
    plt.show()
