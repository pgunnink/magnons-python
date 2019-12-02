import numpy as np
from magnons.amplitudes import AkBk

print('test')
i = 0


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