import numpy as np
from numpy import pi
from scipy.special import erfc
import sys
import warnings


# define helper functions
def f(p, q):
    # res = 0
    # plus_erf = erfc(p + q)
    # if plus_erf > sys.float_info.min:
    #     res += plus_erf * np.exp(2 * p * q)
    # min_erf = erfc(p - q)
    # if min_erf > sys.float_info.min:
    #     res += min_erf * np.exp(-2 * p * q)
    # return res
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        res = np.zeros_like(p)
        plus_erf = erfc(p + q)
        idx = plus_erf > sys.float_info.min
        res[idx] += (plus_erf * np.exp(2 * p * q))[idx]
        min_erf = erfc(p - q)
        idx = min_erf > sys.float_info.min
        res[idx] += (min_erf * np.exp(-2 * p * q))[idx]
        return res


def misra(x):
    return np.exp(-x) * (3 + 2 * x) / (2 * x**2) + 3 * np.sqrt(np.pi) * erfc(
        np.sqrt(x)) / (4 * x**(5 / 2))


class Dk:
    flip = 1

    def __init__(self, eps=None, a=None, mu=None, Nr=4, Ng=4):
        self.eps = eps
        self.Nr = Nr
        self.Ng = Ng
        self.a = a
        self.mu = mu

    def run(self, x, ky, kz):
        real_res = self.real_sum(x, ky, kz)
        recip_res = self.recip_sum(x, ky, kz)
        res = real_res + recip_res
        # for convenience, this returns a float if only run for one x value, else
        # it returns a list. However, almost always you will request a list of x values
        if isinstance(x, float) or isinstance(x, int):
            return res[0]
        else:
            return res

    def real_f(self, x, y, z, ky, kz):
        raise NotImplementedError

    def recip_f(self, p, q, ky, kz, gy, gz):
        raise NotImplementedError

    def real_sum_explicit(self, x, ky, kz):
        res = 0
        for i in range(-self.Nr, self.Nr + 1):
            z = i * self.a
            for j in range(-self.Nr, self.Nr + 1):
                y = j * self.a
                if x == 0 and y == 0 and z == 0:
                    continue
                res += self.real_f(x, y, z, ky, kz)
        return res

    def real_sum(self, x, ky, kz):
        if isinstance(x, float) or isinstance(x, int):
            x = np.ones(1) * x
            Nx = 1
        else:
            Nx = len(x)
        map_y, map_z = np.mgrid[-self.Nr:self.Nr + 1, -self.Nr:self.Nr
                                + 1] * self.a
        # we need this trick to eliminate the x==y==z==0 grid point,
        # leaving it in will gives NaN warnings
        map_y = map_y[:, :, np.newaxis]
        map_z = map_z[:, :, np.newaxis]
        map_y = np.repeat(map_y, Nx, axis=-1)
        map_z = np.repeat(map_z, Nx, axis=-1)
        x = x[np.newaxis, np.newaxis, :]
        x = np.repeat(np.repeat(x, self.Nr * 2 + 1, axis=0),
                      self.Nr * 2 + 1,
                      axis=1)
        idx = (x == 0) & (map_y == 0) & (map_z == 0)
        x[idx] = 1
        res = self.real_f(x, map_y, map_z, ky, kz)
        res[idx] = 0
        return np.sum(res, axis=(0, 1))

    def recip_sum(self, x, ky, kz):
        if isinstance(x, float) or isinstance(x, int):
            x = np.ones(1) * x
            Nx = 1
        else:
            Nx = len(x)

        q = np.sqrt(self.eps) * x
        map_gy, map_gz = np.mgrid[-self.Ng:self.Ng + 1, -self.Ng:self.Ng
                                  + 1] * (2 * np.pi / self.a)
        map_gy = map_gy[:, :, np.newaxis]
        map_gz = map_gz[:, :, np.newaxis]
        map_gy = np.repeat(map_gy, Nx, axis=-1)
        map_gz = np.repeat(map_gz, Nx, axis=-1)
        q = q[np.newaxis, np.newaxis, :]
        q = np.repeat(np.repeat(q, self.Ng * 2 + 1, axis=0),
                      self.Ng * 2 + 1,
                      axis=1)
        p = np.sqrt((ky + map_gy)**2
                    + (kz + map_gz)**2) / (2 * np.sqrt(self.eps))
        return np.sum(self.recip_f(p, q, ky, kz, map_gy, map_gz), axis=(0, 1))

    def recip_sum_explicit(self, x, ky, kz):
        res = 0
        q = np.sqrt(self.eps) * x
        for m in range(-self.Ng, self.Ng + 1):
            gy = 2 * pi * m / self.a
            for n in range(-self.Ng, self.Ng + 1):
                gz = 2 * pi * n / self.a
                p = np.sqrt((ky + gy)**2
                            + (kz + gz)**2) / (2 * np.sqrt(self.eps))
                res += self.recip_f(p, q, ky, kz, gy, gz)
        return res

    def table(self, ky, kz, N):
        # temp = np.array(
        #     [self.run(i * self.a, ky, kz) for i in range(-N + 1, 1)],
        #     dtype=np.complex)
        temp = self.run(np.arange(-N + 1, 1, 1) * self.a, ky, kz)
        return np.concatenate((temp, self.flip * np.flip(temp[:-1])))


class Dkxx(Dk):
    flip = 1

    def recip_f(self, p, q, ky, kz, gy, gz):
        res = (8 * np.sqrt(self.eps)) / (3 * np.sqrt(pi)) * np.exp(-p**2
                                                                   - q**2)
        res -= p * 2 * np.sqrt(self.eps) * f(p, q)
        return -1 * pi * self.mu**2 / (self.a**2) * res

    def real_f(self, x, y, z, ky, kz):
        r2 = y**2 + z**2 + x**2
        return (-4 * self.eps**(5 / 2) * self.mu**2 / (3 * np.sqrt(np.pi)) *
                (r2 - 3 * x**2) * np.cos(ky * y) * np.cos(kz * z)
                * misra(r2 * self.eps))


class Dkyy(Dk):
    flip = 1

    def recip_f(self, p, q, ky, kz, gy, gz):
        res = (4 * np.sqrt(self.eps)) / (3 * np.sqrt(pi)) * np.exp(-p**2
                                                                   - q**2)
        res -= (ky + gy)**2 / (p * 2 * np.sqrt(self.eps)) * f(p, q)
        return pi * self.mu**2 / (self.a**2) * res

    def real_f(self, x, y, z, ky, kz):
        r2 = y**2 + z**2 + x**2
        return (-4 * self.eps**(5 / 2) * self.mu**2 / (3 * np.sqrt(np.pi)) *
                (r2 - 3 * y**2) * np.cos(ky * y) * np.cos(kz * z)
                * misra(r2 * self.eps))


class Dkzz(Dkyy):
    def run(self, x, ky, kz):
        return super().run(x, kz, ky)


class Dkxy(Dk):
    flip = -1

    def recip_f(self, p, q, ky, kz, gy, gz):
        res = (ky + gy) * f(p, q)
        return (1j * pi * self.mu**2 / (self.a**2)
                * np.sign(q / np.sqrt(self.eps)) * res)

    def real_f(self, x, y, z, ky, kz):
        r2 = y**2 + z**2 + x**2
        return (4j * self.eps**(5 / 2) * self.mu**2 / (np.sqrt(np.pi)) * x
                * np.sin(ky * y) * np.cos(kz * z) * misra(r2 * self.eps))


class Dkxz(Dkxy):
    flip = -1

    def run(self, x, ky, kz):
        return super().run(x, kz, ky)


class Dkyz(Dk):
    flip = 1

    def recip_f(self, p, q, ky, kz, gy, gz):
        return -pi * self.mu**2 / (2 * np.sqrt(self.eps) * self.a**2) * (
            (ky + gy) * (kz + gz)) / p * f(p, q)

    def real_f(self, x, y, z, ky, kz):
        r2 = y**2 + z**2 + x**2
        return (4 * self.eps**(5 / 2) * self.mu**2 / (np.sqrt(np.pi)) * y * z
                * np.sin(ky * y) * np.sin(kz * z) * misra(r2 * self.eps))


if __name__ == "__main__":
    from magnons.yig import a, mu

    H = 700.0
    h = mu * H
    eps = a**(-2)

    # target:
    # print(f"Target: -9.91297*10^-21, result: {K.run(0, 10**5, 10**5)}")
    # print(
    #     f"Positive: {K.run(-10*a, 10**5, 10**5)}, Negative: {K.run(10*a, 10**5, 10**5)}"
    # )
    # print(f"zero: {K.run(0, 10**5, 10**5)}")
    # print(K.table(10**4, 10**4, 10))

    K = Dkyy(eps, a, mu, Ng=10, Nr=10)
    print(K.run(0, 10**5, 10**2))
    K = Dkzz(eps, a, mu, Ng=10, Nr=10)
    print(K.run(0, 10**5, 10**2))
