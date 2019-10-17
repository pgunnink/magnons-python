from magnons.energies import get_dispersion_theta
from magnons.yig import a, S, mu, J
from magnons.cgs import E_to_GHz
import matplotlib.pyplot as plt
import numpy as np
if __name__ == "__main__":
    kwargs = {
        "eps": a**-2,
        "a": a,
        "S": S,
        "mu": mu,
        "J": J,
        "E_to_GHz": E_to_GHz,
        "h": mu * 700,
        "Nr": 4,
        "Ng": 4,
    }
    kwargs["N"] = 200
    res, kvalues = get_dispersion_theta(0,
                                        10,
                                        phi=0,
                                        use_angled_if_zero=True,
                                        **kwargs)
    absk = np.sqrt(np.sum(kvalues**2, axis=1))

    for i in range(6):
        plt.semilogx(absk, res[:, i], "-", color="black")
    plt.savefig('./plots/with_BkAkAngle.png')
    plt.figure()

    res, kvalues = get_dispersion_theta(0,
                                        10,
                                        phi=0,
                                        use_angled_if_zero=False,
                                        **kwargs)
    absk = np.sqrt(np.sum(kvalues**2, axis=1))
    for i in range(6):
        plt.semilogx(absk, res[:, i], "-", color="black")
    plt.savefig('./plots/with_BkAk.png')