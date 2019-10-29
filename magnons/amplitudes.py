from magnons.dipolar_sums import Dkzz, Dkxx, Dkyy, Dkxy, Dkyz, Dkxz
import numpy as np


def Jk(i, j, ky, kz, N=None, a=None, J=None):
    if i == j:
        res = J * (6 - 2 * np.cos(ky * a) - 2 * np.cos(kz * a))
        if i == 0:
            res -= J
        if i == N - 1:
            res -= J
        return res
    elif i == j + 1 or i == j - 1:
        return -J
    else:
        return 0


# def AkBkAngleExplicit(ky,
#               kz,
#               phi,
#               alpha,
#               N=None,
#               J=None,
#               S=None,
#               h=None,
#               eps=None,
#               a=None,
#               mu=None,
#               Nr=4,
#               Ng=4):
#     xx = Dkxx(eps=eps, a=a, mu=mu, Nr=Nr, Ng=Ng)
#     yy = Dkyy(eps=eps, a=a, mu=mu, Nr=Nr, Ng=Ng)
#     zz = Dkzz(eps=eps, a=a, mu=mu, Nr=Nr, Ng=Ng)
#     xy = Dkxy(eps=eps, a=a, mu=mu, Nr=Nr, Ng=Ng)
#     xz = Dkxz(eps=eps, a=a, mu=mu, Nr=Nr, Ng=Ng)
#     yz = Dkyz(eps=eps, a=a, mu=mu, Nr=Nr, Ng=Ng)

#     zz_table0 = zz.table(10**-6, 10**-6, N)
#     xx_table0 = xx.table(10**-6, 10**-6, N)
#     xz_table0 = xz.table(10**-6, 10**-6, N)

#     xx_table = xx.table(ky, kz, N)
#     yy_table = yy.table(ky, kz, N)
#     xy_table = xy.table(ky, kz, N)
#     zz_table = zz.table(ky, kz, N)
#     xz_table = xz.table(ky, kz, N)
#     yz_table = yz.table(ky, kz, N)

#     A = np.zeros((N, N), dtype=np.complex)
#     B = np.zeros((N, N), dtype=np.complex)

#     for i in range(N):
#         for j in range(N):
#             if i == j:
#                 A[i, j] = h * np.cos(phi - alpha) + S * np.sum(
#                     zz_table0[i:i + N] * np.cos(phi)**2
#                     + xx_table0[i:i + N] * np.sin(phi)**2
#                     + xz_table0[i:i + N] * np.sin(phi) * np.cos(phi))
#             A[i, j] += S * Jk(i, j, ky, kz, N=N, a=a, J=J)
#             A[i, j] -= (
#                 S / 2 * (xx_table[i - j + N - 1] * np.cos(phi)**2
#                          + yy_table[i - j + N - 1]
#                          + zz_table[i - j + N - 1] * np.sin(phi)**2)
#                 - 2 * xz_table[i - j + N - 1] * np.sin(phi) * np.cos(phi))

#             B[i, j] = -0.5 * S * (
#                 xx_table[i - j + N - 1] * np.cos(phi)**2
#                 - yy_table[i - j + N - 1]
#                 + zz_table[i - j + N - 1] * np.sin(phi)**2
#                 - 2 * xz_table[i - j + N - 1] * np.sin(phi) * np.cos(phi)
#                 + 2j * xy_table[i - j + N - 1] * np.cos(phi)
#                 - 2j * yz_table[i - j + N - 1] * np.sin(phi))
#     return A, B


def AkBkAngle(ky,
              kz,
              phi,
              alpha,
              N=None,
              J=None,
              S=None,
              h=None,
              eps=None,
              a=None,
              mu=None,
              Nr=4,
              Ng=4):
    xx = Dkxx(eps=eps, a=a, mu=mu, Nr=Nr, Ng=Ng)
    yy = Dkyy(eps=eps, a=a, mu=mu, Nr=Nr, Ng=Ng)
    zz = Dkzz(eps=eps, a=a, mu=mu, Nr=Nr, Ng=Ng)
    xy = Dkxy(eps=eps, a=a, mu=mu, Nr=Nr, Ng=Ng)
    xz = Dkxz(eps=eps, a=a, mu=mu, Nr=Nr, Ng=Ng)
    yz = Dkyz(eps=eps, a=a, mu=mu, Nr=Nr, Ng=Ng)

    zz_table0 = zz.table(10**-6, 10**-6, N)
    xx_table0 = xx.table(10**-6, 10**-6, N)
    xz_table0 = xz.table(10**-6, 10**-6, N)

    xx_table = xx.table(ky, kz, N)
    yy_table = yy.table(ky, kz, N)
    xy_table = xy.table(ky, kz, N)
    zz_table = zz.table(ky, kz, N)
    xz_table = xz.table(ky, kz, N)
    yz_table = yz.table(ky, kz, N)

    Atemp = np.zeros((N, N), dtype=np.complex)
    Btemp = np.zeros((N, N), dtype=np.complex)

    Atemp += np.diag([
        h * np.cos(phi - alpha)
        + S * np.sum(zz_table0[i:i + N] * np.cos(phi)**2
                     + xx_table0[i:i + N] * np.sin(phi)**2
                     + xz_table0[i:i + N] * np.sin(phi) * np.cos(phi))
        for i in range(N)
    ])

    Atemp += np.diag(
        np.ones(N) * S * J * (6 - 2 * np.cos(ky * a) - 2 * np.cos(kz * a)))
    Atemp[0, 0] -= S * J
    Atemp[N - 1, N - 1] -= S * J
    Atemp += np.diag(np.ones(N - 1), -1) * -J * S
    Atemp += np.diag(np.ones(N - 1), 1) * -J * S

    for i in range(N):
        Atemp[i, :] -= np.flip(
            S / 2 * (xx_table[i:i + N] * np.cos(phi)**2 + yy_table[i:i + N]
                     + zz_table[i:i + N] * np.sin(phi)**2)
            - 2 * xz_table[i:i + N] * np.sin(phi) * np.cos(phi))
        Btemp[i, :] -= 0.5 * S * np.flip(
            xx_table[i:i + N] * np.cos(phi)**2 - yy_table[i:i + N]
            + zz_table[i:i + N] * np.sin(phi)**2 - 2 * xz_table[i:i + N]
            * np.sin(phi) * np.cos(phi) + 2j * xy_table[i:i + N] * np.cos(phi)
            - 2j * yz_table[i:i + N] * np.sin(phi))
    return Atemp, Btemp


def AkBkAngleExplicit(ky,
                      kz,
                      phi,
                      alpha,
                      N=None,
                      J=None,
                      S=None,
                      h=None,
                      eps=None,
                      a=None,
                      mu=None,
                      Nr=4,
                      Ng=4):
    xx = Dkxx(eps=eps, a=a, mu=mu, Nr=Nr, Ng=Ng)
    yy = Dkyy(eps=eps, a=a, mu=mu, Nr=Nr, Ng=Ng)
    zz = Dkzz(eps=eps, a=a, mu=mu, Nr=Nr, Ng=Ng)
    xy = Dkxy(eps=eps, a=a, mu=mu, Nr=Nr, Ng=Ng)
    xz = Dkxz(eps=eps, a=a, mu=mu, Nr=Nr, Ng=Ng)
    yz = Dkyz(eps=eps, a=a, mu=mu, Nr=Nr, Ng=Ng)

    zz_table0 = zz.table(10**-6, 10**-6, N)
    xx_table0 = xx.table(10**-6, 10**-6, N)
    xz_table0 = xz.table(10**-6, 10**-6, N)

    xx_table = xx.table(ky, kz, N)
    yy_table = yy.table(ky, kz, N)
    xy_table = xy.table(ky, kz, N)
    zz_table = zz.table(ky, kz, N)
    xz_table = xz.table(ky, kz, N)
    yz_table = yz.table(ky, kz, N)

    A = np.zeros((N, N), dtype=np.complex)
    B = np.zeros((N, N), dtype=np.complex)

    for i in range(N):
        for j in range(N):
            if i == j:
                A[i, j] = h * np.cos(phi - alpha) + S * np.sum(
                    zz_table0[i:i + N] * np.cos(phi)**2
                    + xx_table0[i:i + N] * np.sin(phi)**2
                    + xz_table0[i:i + N] * np.sin(phi) * np.cos(phi))
            A[i, j] += S * Jk(i, j, ky, kz, N=N, a=a, J=J)
            A[i, j] -= (
                S / 2 * (xx_table[i - j + N - 1] * np.cos(phi)**2
                         + yy_table[i - j + N - 1]
                         + zz_table[i - j + N - 1] * np.sin(phi)**2)
                - 2 * xz_table[i - j + N - 1] * np.sin(phi) * np.cos(phi))

            B[i, j] = -0.5 * S * (
                xx_table[i - j + N - 1] * np.cos(phi)**2
                - yy_table[i - j + N - 1]
                + zz_table[i - j + N - 1] * np.sin(phi)**2
                - 2 * xz_table[i - j + N - 1] * np.sin(phi) * np.cos(phi)
                + 2j * xy_table[i - j + N - 1] * np.cos(phi)
                - 2j * yz_table[i - j + N - 1] * np.sin(phi))
    return A, B


def AkBk(ky,
         kz,
         N=None,
         J=None,
         S=None,
         h=None,
         eps=None,
         a=None,
         mu=None,
         Nr=4,
         Ng=4):
    xx = Dkxx(eps=eps, a=a, mu=mu, Nr=Nr, Ng=Ng)
    yy = Dkyy(eps=eps, a=a, mu=mu, Nr=Nr, Ng=Ng)
    zz = Dkzz(eps=eps, a=a, mu=mu, Nr=Nr, Ng=Ng)
    xy = Dkxy(eps=eps, a=a, mu=mu, Nr=Nr, Ng=Ng)

    xx_table = xx.table(ky, kz, N)
    yy_table = yy.table(ky, kz, N)
    zz_table0 = zz.table(10**-6, 10**-6, N)
    xy_table = xy.table(ky, kz, N)

    Atemp = np.zeros((N, N), dtype=np.complex)
    Btemp = np.zeros((N, N), dtype=np.complex)
    Atemp += np.diag([h + S * np.sum(zz_table0[i:i + N]) for i in range(N)])
    Atemp += np.diag(
        np.ones(N) * S * J * (6 - 2 * np.cos(ky * a) - 2 * np.cos(kz * a)))
    Atemp[0, 0] -= S * J
    Atemp[N - 1, N - 1] -= S * J
    Atemp += np.diag(np.ones(N - 1), -1) * -J * S
    Atemp += np.diag(np.ones(N - 1), 1) * -J * S
    for i in range(N):
        Atemp[i, :] -= .5 * S * np.flip(xx_table[i:i + N] + yy_table[i:i + N])
        Btemp[i, :] -= .5 * S * np.flip(xx_table[i:i + N]
                                        - 2j * xy_table[i:i + N]
                                        - yy_table[i:i + N])
    return Atemp, Btemp


def AkBkExplicit(ky,
                 kz,
                 N=None,
                 J=None,
                 S=None,
                 h=None,
                 eps=None,
                 a=None,
                 mu=None,
                 Nr=4,
                 Ng=4):
    xx = Dkxx(eps=eps, a=a, mu=mu, Nr=Nr, Ng=Ng)
    yy = Dkyy(eps=eps, a=a, mu=mu, Nr=Nr, Ng=Ng)
    zz = Dkzz(eps=eps, a=a, mu=mu, Nr=Nr, Ng=Ng)
    xy = Dkxy(eps=eps, a=a, mu=mu, Nr=Nr, Ng=Ng)

    xx_table = xx.table(ky, kz, N)
    yy_table = yy.table(ky, kz, N)
    zz_table0 = zz.table(10**-6, 10**-6, N)
    xy_table = xy.table(ky, kz, N)

    Atemp = np.zeros((N, N), dtype=np.complex)
    Btemp = np.zeros((N, N), dtype=np.complex)
    for i in range(N):
        for j in range(N):
            if i == j:
                Atemp[i, j] = h + S * np.sum(zz_table0[i:i + N])
            Atemp[i, j] += S * Jk(i, j, ky, kz, N=N, a=a, J=J)
            Atemp[i, j] -= S / 2 * (xx_table[i - j + N - 1]
                                    + yy_table[i - j + N - 1])

            Btemp[i, j] = -0.5 * S * (xx_table[i - j + N - 1]
                                      - 2j * xy_table[i - j + N - 1]
                                      - yy_table[i - j + N - 1])
    return Atemp, Btemp
