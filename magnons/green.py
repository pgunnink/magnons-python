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
        damping = -np.diag(np.ones(self.N) * self.damping)
        damping[0] -= self.damping_IF
        omega_diag = np.diag(np.ones(self.N) * omega)
        G_inv = np.block([[A + damping + omega_diag, B],
                          [-B.T, -A + damping - omega_diag]])
        driving = self.driving * np.ones(self.N * 2)

        b = np.linalg.solve(G_inv, driving)
        return b

    def current_per_k(self):
        E, k = self.get_lowest_E()
        res = []
        for k_i, E_i in zip(k, E):
            omega = E_i / self.E_to_GHz
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
        return E[:, 0], k

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
        ham_Sz = .5j * (A @ (b[:self.N].conj() * b[:self.N]
                             + b[-self.N:].conj() * b[-self.N:])
                        - B.T @ (b[-self.N:].conj() * b[:self.N])
                        + B @ (b[:self.N].conj() * b[-self.N:]))
        ham_Sz = ham_Sz.imag
        Sz_damp = np.imag(self.S * self.damping * omega * b[:self.N]
                          * b[-self.N:])
        Sz_damp[0] += np.imag(self.S * self.damping_IF * omega * b[0]
                              * b[self.N])
        Sz_driving = self.S * np.imag(self.driving * b[:self.N])
        print(np.sum(ham_Sz), np.sum(Sz_damp), np.sum(Sz_driving))
        return ham_Sz + Sz_damp + Sz_driving

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
        res = self.S * (self.damping
                        + self.damping_IF) * omega * b[0] * b[self.N]
        return res