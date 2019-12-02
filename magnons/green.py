import numpy as np
from magnons.amplitudes import AkBk


def G(
    ky,
    kz,
    omega,
    alpha=None,
    alpha_IF=None,
    N=None,
    J=None,
    S=None,
    h=None,
    eps=None,
    a=None,
    mu=None,
):
    A, B = AkBk(ky, kz, N=N, J=J, S=S, h=h, eps=eps, a=a, mu=mu)
    damping = -np.diag(np.ones(N) * alpha)
    damping[0] -= alpha_IF
    omega_diag = np.diag(np.ones(N) * omega)
    G_inv = np.block([[A + damping + omega_diag, B],
                      [-B.T, -A + damping - omega_diag]])
    driving = np.ones(N)

    b = np.linalg.solve(G_inv, driving)
    return b


def get_current_in_lead(ky,
                        kz,
                        omega,
                        alpha=None,
                        alpha_IF=None,
                        N=None,
                        J=None,
                        S=None,
                        h=None,
                        eps=None,
                        a=None,
                        mu=None):
    b = G(ky,
          kz,
          omega,
          alpha=alpha,
          alpha_IF=alpha_IF,
          N=N,
          J=J,
          S=S,
          h=h,
          eps=eps,
          a=a,
          mu=mu)
    res = S * (alpha + alpha_IF) * omega * b[0].conj() * b[0]
    return res