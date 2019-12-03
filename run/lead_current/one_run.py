from magnons.green import Green
from magnons.yig import S, mu, a, J, Ms
from magnons.cgs import E_to_GHz
from magnons.magnetization import magnetization_angle
import numpy as np
import matplotlib.pyplot as plt
if __name__ == "__main__":
    ky = 10**-2
    kz = 10**4
    omega = 1 / E_to_GHz
    damping = 0.001
    damping_IF = 1
    N = 100
    eps = a**(-2)
    H = 2200
    h = H * mu
    res = []
    alpha = 30
    phi = magnetization_angle(np.radians(alpha), M=Ms, H=H)
    G = Green(theta=0,
              phi=phi,
              alpha=np.radians(alpha),
              damping=damping,
              damping_IF=damping_IF,
              N=N,
              J=J,
              S=S,
              h=h,
              eps=eps,
              a=a,
              mu=mu,
              E_to_GHz=E_to_GHz)
    res = G.get_Sz(10**-2, 10**4, 8 / E_to_GHz)
    print(np.sum(res))
