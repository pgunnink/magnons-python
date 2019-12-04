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
    damping = 5 * 10**(-3)
    damping_IF = 1
    N = 100
    eps = a**(-2)
    H = 2200
    h = H * mu
    res = []
    for alpha in [0, 10, 30, 60, 80]:
        phi = magnetization_angle(np.radians(alpha), M=Ms, H=H)
        G = Green(Nk=4,
                  theta=np.radians(0),
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
                  E_to_GHz=E_to_GHz,
                  driving=h / 100)
        res.append(G.current_per_k())
    res = np.array(res)
    plt.plot(res.T)
    plt.show()
