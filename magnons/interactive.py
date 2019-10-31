import numpy as np
import matplotlib.pyplot as plt
from magnons.data import Data
from magnons.energies import ev_in_HP_basis
from magnons.spin import get_spincurrent


class DoublePlot:
    def __init__(self, kvalues, energies, ev):
        self.kvalues = kvalues
        self.kabs = np.sqrt(np.sum(kvalues**2, axis=1))
        self.energies = energies
        self.ev = ev

    def plot_E(self, Nlim=6, logplot=True, ylim=None):
        self.fig, (self.ax_E, self.ax_ev) = plt.subplots(1, 2)
        for i in range(Nlim):
            if logplot:
                self.ax_E.semilogx(self.kabs,
                                   self.energies[:, i],
                                   '*-',
                                   color='black')
            else:
                self.ax_E.plot(
                    self.kabs,
                    self.energies[:, i],
                    '-',
                    color='black',
                )
        if ylim is not None:
            self.ax_E.set_ylim(ylim)
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.selected_point = None

    def plot_ev(self, k_i, E_i):
        ev = self.ev[k_i, :, E_i]
        ev = ev_in_HP_basis(ev)
        print(np.sum(ev))
        self.ax_ev.clear()
        self.ax_ev.plot(np.real(ev), label='Re', color='red')
        self.ax_ev.plot(np.imag(ev), label='Im', color='blue')
        self.ax_ev.legend()

    def onclick(self, event):
        # print(f'Edata: {event.ydata}, kdata: {event.xdata}')
        # first find closest k point
        k_i = (np.abs(self.kabs - event.xdata)).argmin()
        E_i = (np.abs(self.energies[k_i, :] - event.ydata)).argmin()

        k = self.kabs[k_i]
        E = self.energies[k_i, E_i]
        # print(
        #     f"found E {self.energies[k_i, E_i]}, found k {np.sqrt(np.sum(self.kvalues[k_i]**2))}"
        # )

        if self.selected_point is None:
            self.selected_point, = self.ax_E.plot(k,
                                                  E,
                                                  'X',
                                                  color='red',
                                                  markersize=12)
        else:
            self.selected_point.set_xdata(k)
            self.selected_point.set_ydata(E)
        self.plot_ev(k_i, E_i)
        self.fig.canvas.draw()


class DoubePlotSpinCurrent(DoublePlot):
    def plot_ev(self, k_i, E_i):
        ev = self.ev[k_i, :, E_i]
        ev = ev_in_HP_basis(ev)
        spin_current = np.real(get_spincurrent(ev))
        print(np.sum(spin_current))
        self.ax_ev.clear()
        self.ax_ev.plot(spin_current)


class DoubePlotFourier(DoublePlot):
    def plot_ev(self, k_i, E_i):
        ev = self.ev[k_i, :, E_i]
        ev = ev_in_HP_basis(ev)
        sp = np.fft.fft(ev)
        freq = np.fft.fftfreq(len(ev))

        self.ax_ev.clear()
        self.ax_ev.plot(freq, sp.real, label=f'Re {sp.real[0]:.2e}')
        self.ax_ev.plot(freq, sp.imag, label=f"Im {sp.imag[0]:.2e}")
        self.ax_ev.legend()