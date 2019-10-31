import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from magnons.energies import klist_ev_in_HP_basis


def get_uniform_component(ev):
    Nk = ev.shape[0]
    N = ev.shape[2]
    if ev.shape[1] != ev.shape[2]:
        ev = klist_ev_in_HP_basis(ev)
    uni_comp = np.zeros((Nk, N))
    for ik in range(Nk):
        for i in range(N):
            sp = np.fft.fft(ev[ik, :, i])
            uni_comp[ik, i] = np.abs(sp[0])
    return uni_comp


def get_spincurrent(ev, S=1, J=1):
    if len(ev.shape) > 1:
        raise Exception('only compatible with shape (N,)')
    ev_conj = ev.conj()  # is faster this way
    current = -1j * ((ev_conj[1:] * ev[:-1]) - ev_conj[:-1] * (ev[1:]))
    return current


def get_spincurrent_dispersion(ev, S=1, J=1):
    # assuming ev has shape (Nk, 2N, N)
    Nk = ev.shape[0]
    N = ev.shape[2]
    if ev.shape[1] != ev.shape[2]:
        ev = klist_ev_in_HP_basis(ev)
    current = np.zeros((Nk, N))
    for ik in range(Nk):
        for i in range(N):
            current[ik, i] = np.average(
                np.real(get_spincurrent(ev[ik, :, i], S=S, J=J)))
    return current


def get_spincurrent_total(ev, S=1, J=1):
    current = get_spincurrent_dispersion(ev, S=S, J=J)
    return np.sum(current, axis=1)


def plot_spincurrent_dispersion(kvalues,
                                E,
                                ev,
                                norm=None,
                                current=None,
                                S=1,
                                J=1,
                                ax=None,
                                cmap=None):
    if current is None:
        current = get_spincurrent_dispersion(ev, S=S, J=J)
    if norm is not None:
        norm = plt.Normalize(current.min(), current.max())
    # current = norm(current)
    # current = current / current.max()
    kabs = np.sqrt(np.sum(kvalues**2, axis=1))
    if cmap is None:
        cmap = mpl.cm.get_cmap('cool')

    if ax is None:
        fig, ax = plt.subplots()
        divider = make_axes_locatable(ax)
        ax_cb = divider.append_axes('right', size='5%', pad=0.05)
        mpl.colorbar.ColorbarBase(ax_cb,
                                  cmap=cmap,
                                  norm=norm,
                                  orientation='vertical')

    for i in range(E.shape[-1]):
        y_points = E[:, i]
        c = current[:, i]
        points = np.array([kabs, y_points]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(c)
        ax.add_collection(lc)

    ax.set_xlim(kabs.min(), kabs.max())
    ax.set_ylim(E.min() * .95, E.min() + 10)
