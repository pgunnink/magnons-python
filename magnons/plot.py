import matplotlib.pyplot as plt
import numpy as np
from magnons.spin import get_spincurrent_total
from tqdm import tqdm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable


def create_edges(x, upper_lim=None, lower_lim=0):
    N = len(x)
    new_x = []
    for i in range(N):
        if i == 0:
            diff = x[1] - x[0]
        else:
            diff = x[i] - x[i - 1]
        new_x.append(x[i] - .5 * diff)
    new_x.append(x[-1] + .5 * diff)

    if lower_lim is not None and new_x[0] < lower_lim:
        new_x[0] = 0
    if upper_lim is not None and new_x[-1] > upper_lim:
        new_x[-1] = upper_lim
    return new_x


def add_colorbar_subplots(fig, cmap, norm=None):
    if norm is None:
        norm = plt.Normalize(0, 1)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                 ax=fig.get_axes())


def add_colorbar_oneplot(ax, cmap, norm):
    divider = make_axes_locatable(ax)
    ax_cb = divider.append_axes('right', size='5%', pad=0.05)
    mpl.colorbar.ColorbarBase(ax_cb,
                              cmap=cmap,
                              norm=norm,
                              orientation='vertical')


def plot_colored_dispersion(k, E, colors, ax=None, norm=None, cmap='jet'):
    if isinstance(cmap, str):
        cmap = mpl.cm.get_cmap(cmap)
    if norm is None:
        norm = plt.Normalize(colors.min(), colors.max())
    if ax is None:
        # create axes and add a colorbar
        fig, ax = plt.subplots()
        add_colorbar_oneplot(ax, cmap, norm)
    for i in range(E.shape[-1]):
        y_points = E[:, i]
        c = colors[:, i]
        points = np.array([k, y_points]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = mpl.collections.LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(c)
        ax.add_collection(lc)
    ax.set_xlim(k.min(), k.max())
    ax.set_ylim(E.min(), E.max())
    return ax


def plot_totalspincurrent(process, S=1, J=1):
    values = list(process.get_all())
    current = [get_spincurrent_total(x[2], J=J, S=S) for x in tqdm(values)]
    current = np.array(current)
    current /= np.max(np.abs(current))
    alpha = np.array([x[3]['alpha'] for x in values])
    phi = np.array([x[3]['phi'] for x in values])
    kvalues = np.array([x[0] for x in values])
    kvalues *= 10**(-4)  # convert to micrometer^-1
    kabs = np.sqrt(np.sum(kvalues[0]**2, axis=1))

    sort_idx = np.argsort(phi)

    alpha = alpha[sort_idx]
    phi = phi[sort_idx]
    current = np.array(current)[sort_idx]
    alpha = create_edges(alpha, upper_lim=90)
    phi = create_edges(phi, upper_lim=90)
    kabs = create_edges(kabs)
    X, Y = np.meshgrid(kabs, phi)
    plt.pcolormesh(X, Y, current, cmap='jet')
    plt.colorbar()
    plt.xlabel('|k|')
    plt.ylabel('Internal Magnetization angle')

    plt.show()
