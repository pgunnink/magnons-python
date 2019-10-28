import os

import matplotlib.pyplot as plt
import numpy as np
from magnons.process import Process

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    plt.figure(dpi=200, figsize=(16, 8))
    p = Process(dir_path)
    for i, (kvalues, E, ev, attrs) in enumerate(p.get_all()):
        plt.subplot(2, 2, i + 1)
        kabs = np.sqrt(np.sum(kvalues**2, axis=1))
        alpha = attrs['alpha']
        phi = attrs['phi']
        for n in range(50):
            plt.plot(kabs, E[:, n], '-', color='black')
            plt.title(f'Alpha {alpha:.1f}, phi {phi:.1f}')
        plt.ylim(2, 20)
        plt.savefig('temp.png')
    plt.show()
