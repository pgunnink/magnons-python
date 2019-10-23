from magnons.energies import get_dispersion_theta
import numpy as np
from magnons.yig import a, S, mu, J, Ms
from magnons.cgs import E_to_GHz
import matplotlib.pyplot as plt
from magnons.magnetization import magnetization_angle

if __name__ == '__main__':
    N = 400
    H = 2500
    kwargs = {
        "eps": a**-2,
        "a": a,
        "S": S,
        "mu": mu,
        "J": J,
        "E_to_GHz": E_to_GHz,
        "h": mu * H,
        "Nr": 4,
        "Ng": 4,
    }
    theta = np.radians(90)

    kwargs["N"] = N
    firstN = 30
    al = 30
    al = np.radians(al)
    kwargs["alpha"] = al
    phi = magnetization_angle(al, M=Ms, H=H)
    res, _, kvalues = get_dispersion_theta(theta,
                                           12,
                                           ky_begin=1,
                                           ky_end=10**6,
                                           phi=phi,
                                           firstN=firstN,
                                           use_angled_if_zero=True,
                                           logspace=False,
                                           parallel=True,
                                           **kwargs)
    absk = np.sqrt(np.sum(kvalues**2, axis=1))
    for i in range(firstN):
        plt.plot(absk, res[:, i], "-", color="black")
    plt.title(
        f'H angle {np.round(np.degrees(al))}, M angle {np.round(np.degrees(phi))}'
    )
    plt.show()
