import os

import matplotlib.pyplot as plt
import numpy as np

from magnons.process import Process
from magnons.spin import plot_spincurrent_dispersion, get_spincurrent_dispersion
from magnons.yig import J, S

import matplotlib as mpl

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    p = Process(dir_path)
    k = []
    current = []
    energies = []
    evs = []
    title = []
    for i, (kvalues, E, ev, attrs) in enumerate(p.get_all()):
        title.append(
            f"Magnetization angle {attrs['phi']:.0f}, field angle {attrs['alpha']:.0f}"
        )
        current.append(get_spincurrent_dispersion(ev, S=S, J=J))
        k.append(kvalues)
        energies.append(E)
        evs.append(ev)
    current = np.array(current)
    current = current / current.max()
    norm = plt.Normalize(current.min(), current.max())
    cmap = mpl.cm.get_cmap('jet')
    fig = plt.figure()
    for i, (kvalues, E, ev, c) in enumerate(zip(k, energies, evs, current)):
        ax = fig.add_subplot(2, 2, i + 1)
        plot_spincurrent_dispersion(kvalues,
                                    E,
                                    ev,
                                    norm=norm,
                                    current=c,
                                    S=S,
                                    J=J,
                                    ax=ax,
                                    cmap=cmap)
        plt.title(title[i])
        plt.ylim(top=100)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                 ax=fig.get_axes())
    plt.show()
