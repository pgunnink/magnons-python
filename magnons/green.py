import numpy as np
from magnons.amplitudes import AkBkAngle
from magnons.energies import get_dispersion_theta


class Green:
    def __init__(self,
                 Nk=12,
                 theta=0,
                 ky_begin=2,
                 ky_end=6,
                 logspace=True,
                 alpha=0,
                 phi=0,
                 damping=None,
                 damping_IF=None,
                 N=None,
                 J=None,
                 S=None,
                 h=None,
                 eps=None,
                 a=None,
                 mu=None,
                 E_to_GHz=None,
                 driving=1):
        self.Nk = Nk
        self.theta = theta
        self.ky_begin = ky_begin
        self.ky_end = ky_end
        self.logspace = logspace
        self.alpha = alpha
        self.phi = phi
        self.damping = damping
        self.damping_IF = damping_IF
        self.N = N
        self.J = J
        self.S = S
        self.h = h
        self.eps = eps
        self.a = a
        self.mu = mu
        self.E_to_GHz = E_to_GHz
        self.driving = driving

    def get_b(self, ky, kz, omega, A=None, B=None):
        if A is None or B is None:
            A, B = AkBkAngle(ky,
                             kz,
                             self.phi,
                             self.alpha,
                             N=self.N,
                             J=self.J,
                             S=self.S,
                             h=self.h,
                             eps=self.eps,
                             a=self.a,
                             mu=self.mu)
        damping = np.diag(np.ones(self.N) * self.damping)
        damping[0] += self.damping_IF
        omega_diag = np.diag(np.ones(self.N) * omega)
        G_inv = np.block([[-A - 1j * damping - omega_diag, -B],
                          [B.T.conj(), A - 1j * damping + omega_diag]])
        driving = self.driving * np.ones(self.N * 2)

        b = np.linalg.solve(G_inv, driving)
        return b

    def current_per_k(self):
        E, k = self.get_lowest_E()
        res = []
        for k_i, E_i in zip(k, E):
            omega = 2 * np.pi * E_i  # * 10**9

            res.append(self.get_current_in_lead(k_i[0], k_i[1], omega))
        return res

    def get_lowest_E(self):
        E, _, k = get_dispersion_theta(
            self.theta,
            self.Nk,
            phi=self.phi,
            alpha=self.alpha,
            use_angled_if_zero=True,
            ky_begin=self.ky_begin,
            ky_end=self.ky_end,
            return_eigenfunctions=True,
            N=self.N,
            J=self.J,
            S=self.S,
            h=self.h,
            eps=self.eps,
            a=self.a,
            mu=self.mu,
        )
        return E[:, 1], k

    def get_Sz(self, ky, kz, omega):
        A, B = AkBkAngle(ky,
                         kz,
                         self.phi,
                         self.alpha,
                         N=self.N,
                         J=self.J,
                         S=self.S,
                         h=self.h,
                         eps=self.eps,
                         a=self.a,
                         mu=self.mu)
        b = self.get_b(ky, kz, omega, A=A, B=B)
        A -= np.diag(A)
        Sz_ham = b * (np.block([[A, -B], [B.conj().T, -A]]) @ b.conj())
        Sz_ham = Sz_ham[:self.N]
        # Sz_damp = self.damping * omega * b[:self.N] * b[-self.N:]
        # Sz_damp[0] += self.damping_IF * omega * b[0] * b[self.N]
        damping = np.ones(self.N) * self.damping
        damping[0] += self.damping_IF
        Sz_damp = damping * omega * (
            np.cos(self.phi) * 2 * b[:self.N] * b[-self.N:]
            + np.sqrt(2 * self.S) / 2 * np.sin(self.phi) *
            (b[:self.N] - b[-self.N:]))

        Sz_driving = 1j * np.sqrt(
            2 * self.S) / 2 * self.driving * (b[:self.N] - b[-self.N:])
        print(np.sum(Sz_ham.imag), np.sum(Sz_damp.imag),
              np.sum(Sz_driving.imag))
        print(f"Spin current in lead: {Sz_damp[0]}")
        return Sz_ham + Sz_damp + Sz_driving

    def get_current_in_lead(
        self,
        ky,
        kz,
        omega,
    ):
        b = self.get_b(
            ky,
            kz,
            omega,
        )
        Sz_damp = (self.damping + self.damping_IF) * omega * (
            np.cos(self.phi) * 2 * b[0] * b[self.N]
            + np.sqrt(2 * self.S) / 2 * np.sin(self.phi) * (b[0] - b[self.N]))
        return Sz_damp