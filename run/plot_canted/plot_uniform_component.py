import os

import matplotlib.pyplot as plt
import numpy as np
from magnons.process import Process
from magnons.spin import get_uniform_component
from magnons.plot import plot_colored_dispersion, add_colorbar_subplots
import matplotlib as mpl

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    fig = plt.figure(dpi=150, figsize=(16, 8))
    p = Process(dir_path)
    cmap = 'Purples'
    for i, (kvalues, E, ev, attrs) in enumerate(p.get_all()):
        ax = fig.add_subplot(2, 2, i + 1)
        kabs = np.sqrt(np.sum(kvalues**2, axis=1)) * 10**(-4)  # convert to um
        alpha = attrs['alpha']
        phi = attrs['phi']
        uni_comp = get_uniform_component(ev)
        uni_comp /= np.max(uni_comp)
        plot_colored_dispersion(kabs, E, uni_comp, ax=ax, cmap=cmap)

        ax.set_ylim(E.min() * .95, E.min() + 5)
        ax.set_xlabel(r'$k (\mu m)$')

    add_colorbar_subplots(fig, cmap)
    plt.show()
