import numpy as np
from magnons.dipolar_sums import Dkxx, Dkyy, Dkzz, Dkxy, Dkxz, Dkyz
from magnons.energies import ev_in_HP_basis, ev_HP_to_S
from magnons.amplitudes import Jk, AkBkAngleExplicit


def spin_momentum_linear(ev,
                         E_i,
                         ky,
                         kz,
                         N,
                         phi=None,
                         a=None,
                         mu=None,
                         S=None,
                         J=None,
                         h=None,
                         alpha=None):
    ev[:, -N:] = np.flip(ev[:, -N:], axis=1)  # reflip the stuff
    eps = a**(-2)
    A, B = AkBkAngleExplicit(ky,
                             kz,
                             phi,
                             alpha,
                             N=N,
                             J=J,
                             S=S,
                             h=h,
                             eps=eps,
                             a=a,
                             mu=mu)
    U = ev[:N, :N]
    W = ev[:N:, -N:]
    V = ev[-N:, :N]
    X = ev[-N:, -N:]

    dtbk = 1j / 2 * ((U + W) @ A + (V + X) @ B)
    dtbmink_dagger = -1j / 2 * ((V + X) @ A.T + (U + W) @ B.T.conj())

    x = .5 * (dtbk + dtbmink_dagger)[:, E_i]
    y = -.5j * (dtbk - dtbmink_dagger)[:, E_i]
    z = (dtbk * dtbmink_dagger)[:, E_i]

    return np.stack((x, y, z)).T


def spin_momentum_linear_old1(ev,
                              E_i,
                              ky,
                              kz,
                              N,
                              phi=None,
                              a=None,
                              mu=None,
                              S=None,
                              J=None,
                              h=None,
                              alpha=None):
    ev = ev_in_HP_basis(ev)
    eps = a**(-2)
    xx = Dkxx(eps=eps, a=a, mu=mu)
    yy = Dkyy(eps=eps, a=a, mu=mu)
    zz = Dkzz(eps=eps, a=a, mu=mu)
    xy = Dkxy(eps=eps, a=a, mu=mu)
    xz = Dkxz(eps=eps, a=a, mu=mu)
    yz = Dkyz(eps=eps, a=a, mu=mu)

    zz_table0 = zz.table(np.finfo('float64').eps, np.finfo('float64').eps, N)
    xx_table0 = xx.table(np.finfo('float64').eps, np.finfo('float64').eps, N)
    xz_table0 = xz.table(np.finfo('float64').eps, np.finfo('float64').eps, N)

    xx_table = xx.table(ky, kz, N)
    yy_table = yy.table(ky, kz, N)
    xy_table = xy.table(ky, kz, N)
    zz_table = zz.table(ky, kz, N)
    xz_table = xz.table(ky, kz, N)
    yz_table = yz.table(ky, kz, N)
    temp = []
    for i in range(N):
        bi = ev[i, E_i]
        bimink = ev[i, N * 2 - E_i - 1]
        dt_bi = []
        dt_bi_mink = []
        for j in range(N):
            bj = ev[j, E_i]
            bjmink = ev[j, N * 2 - E_i - 1]
            A = 0
            if i == j:
                A += h * np.cos(phi - alpha) + S * np.sum(
                    zz_table0[i:i + N] * np.cos(phi)**2
                    + xx_table0[i:i + N] * np.sin(phi)**2
                    + xz_table0[i:i + N] * np.sin(phi) * np.cos(phi))
            A += S * Jk(i, j, ky, kz, N=N, a=a, J=J)
            A -= (S / 2 *
                  (xx_table[i - j + N - 1] * np.cos(phi)**2
                   + yy_table[i - j + N - 1]
                   + zz_table[i - j + N - 1] * np.sin(phi)**2
                   - 2 * xz_table[i - j + N - 1] * np.sin(phi) * np.cos(phi)))

            B = -0.5 * S * (
                xx_table[i - j + N - 1] * np.cos(phi)**2
                - yy_table[i - j + N - 1]
                + zz_table[i - j + N - 1] * np.sin(phi)**2
                - 2 * xz_table[i - j + N - 1] * np.sin(phi) * np.cos(phi)
                + 2j * xy_table[i - j + N - 1] * np.cos(phi)
                - 2j * yz_table[i - j + N - 1] * np.sin(phi))
            # if i == j:
            #     dt_bi.append(-1j * (bj * A +
            #                         (.5 * bi.conj() + .5 * bjmink) * B.conj()))
            # else:
            dt_bi.append(-1j * (bj * A + .5 * bjmink * B.conj()))
            dt_bi_mink.append(-1j * (bjmink * A + .5 * bj * B.conj()))
        dt_bi = np.array(dt_bi)
        y = -1j / 2 * (dt_bi - dt_bi.conj()).sum()
        x = 1 / 2 * (dt_bi + dt_bi.conj()).sum()
        z = (dt_bi * dt_bi.conj()).sum()
        temp.append([x, y, z])
    temp = np.array(temp)
    return temp


def spin_momentum_linear_old(ev,
                             ky,
                             kz,
                             N,
                             phi=None,
                             a=None,
                             mu=None,
                             S=None,
                             J=None,
                             h=None,
                             alpha=None):
    ev = ev_in_HP_basis(ev)
    sqrS = .5 * np.sqrt(2 * S)
    eps = a**(-2)

    xx = Dkxx(eps=eps, a=a, mu=mu).table(ky, kz, N)
    yy = Dkyy(eps=eps, a=a, mu=mu).table(ky, kz, N)
    zz = Dkzz(eps=eps, a=a, mu=mu).table(ky, kz, N)
    xy = Dkxy(eps=eps, a=a, mu=mu).table(ky, kz, N)
    xz = Dkxz(eps=eps, a=a, mu=mu).table(ky, kz, N)
    yz = Dkyz(eps=eps, a=a, mu=mu).table(ky, kz, N)
    zz0 = Dkzz(eps=eps, a=a, mu=mu).table(10**(-10), 10**(-10), N)
    Q = np.sin(phi)
    C = np.cos(phi)
    res = []  # first calculate per x_i point
    for i in range(N):
        bi = ev[i]
        temp_res = []
        for j in range(N):
            # delta = i - j + N - 1
            if i != j:
                continue
            bj = ev[j]
            x = 0
            z = 0
            d = i - j + N - 1

            SxSx = S**2 * Q**2 + 2 * sqrS * S * C * Q * (bj + bj.conj())
            SxSy = -1j * sqrS * S * (bj - bj.conj()) * Q
            SxSz = sqrS * S * (bj + bj.conj()) * (C**2 - Q**2) + S**2 * Q * C
            SySy = 0
            SySz = -1j * sqrS * S * (bj - bj.conj()) * C
            SzSz = S**2 * C**2 - 2 * sqrS * S * C * Q * (bj + bj.conj())
            sum_z0 = 0
            for n in range(N):
                sum_z0 += zz0[i - n + N - 1]

            # y = SxSx * xz[d] + SxSy * yz[d] + SxSz * zz[d]
            y = SxSx * xz[d] + SxSy * yz[d] + SxSz * sum_z0
            y -= SxSz * xx[d] + SySz * xy[d] + SzSz * xz[d]

            # y = sqrS * 2 * S * ((xx[d] - np.sum(zz0[i:i + N])) *
            #                     (bi + bi.conj()) + 1j * xy[d] *
            #                     (bi - bi.conj()))
            J0 = S * J * (6 - 2 * (np.cos(ky * a) + np.cos(kz * a)))
            if i == 0:
                J0 -= S * J
            if i == N - 1:
                J0 -= S * J
            # y += J0 * sqrS * (bi + bi.conj()) + h
            temp_res.append([x, y, z])
        temp_res = np.array(temp_res)
        res.append(np.sum(temp_res, axis=0))
    res = np.array(res)
    return res


def spin_momentum_nonlinear(ev, ky, kz, N, a=None, mu=None, S=None):
    ev_HP = ev_in_HP_basis(ev)
    ev_S = ev_HP_to_S(ev_HP, S=S)

    eps = a**(-2)

    xx = Dkxx(eps=eps, a=a, mu=mu).table(ky, kz, N)
    yy = Dkyy(eps=eps, a=a, mu=mu).table(ky, kz, N)
    zz = Dkzz(eps=eps, a=a, mu=mu).table(ky, kz, N)
    xy = Dkxy(eps=eps, a=a, mu=mu).table(ky, kz, N)
    xz = Dkxz(eps=eps, a=a, mu=mu).table(ky, kz, N)
    yz = Dkyz(eps=eps, a=a, mu=mu).table(ky, kz, N)

    res = []  # first calculate per x_i point
    for i in range(N):
        Si = ev_S[:, i]
        temp_res = []
        for j in range(N):
            # delta = i - j + N - 1
            if i != j:
                continue
            Sj = ev_S[:, j]
            Sx = Sj[0]
            Sy = Sj[1]
            Sz = Sj[2]
            x = Sx * xx[i - j + N - 1] + Sy * xy[i - j + N
                                                 - 1] + Sz * xz[i - j + N - 1]
            y = Sy * yy[i - j + N - 1] + Sz * yz[i - j + N
                                                 - 1] + Sx * xy[i - j + N - 1]
            z = Sz * zz[i - j + N - 1] + Sy * yz[i - j + N
                                                 - 1] + Sx * xz[i - j + N - 1]
            G = np.stack((x, y, z))
            temp_res.append(np.cross(G, Si))
        temp_res = np.array(temp_res)
        res.append(np.sum(temp_res, axis=0))
    res = np.array(res)
    return res


def K(k, x, a=None, N=50):
    R = lambda x, y, z: np.sqrt(x**2 + y**2 + z**2)
    exp = lambda y, z, k: np.exp(-1j * (k[0] * y + k[1] * z))

    ylam = lambda x, y, z: -((3 * x * y * y + 3 * x * z * y) / R(x, y, z)**7)
    zlam = lambda x, y, z: -((3 * x * y * z + 3 * x * z * y) / R(x, y, z)**7)

    res_y = 0
    res_z = 0
    for i in range(-N, N + 1):
        y = i * a
        for j in range(-N, N + 1):
            z = j * a
            exp_local = exp(y, z, k)
            res_y += exp_local * ylam(x, y, z)
            res_z += exp_local * zlam(x, y, z)
    return np.array([0, res_y, res_z])


def Gx(k, x, a=None, N=50):
    exp = lambda y, z, k: np.exp(-1j * (k[0] * y + k[1] * z))
    R = lambda x, y, z: np.sqrt(x**2 + y**2 + z**2)
    res = 0
    for i in range(-N, N + 1):
        y = i * a
        for j in range(-N, N + 1):
            z = j * a
            res += exp(y, z, k) * R**(-5)
    return res


if __name__ == '__main__':
    from magnons.yig import a, mu
    print(mu**2 * K([10**4, 10**4], a, a=a))
