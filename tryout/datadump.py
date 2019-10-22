from magnons.energies import get_dispersion_theta
from magnons.data import Data
from magnons.yig import a, S, mu, J
from magnons.cgs import E_to_GHz

if __name__ == "__main__":
    # H = 700
    # kwargs = {
    #     "eps": a**-2,
    #     "a": a,
    #     "S": S,
    #     "mu": mu,
    #     "J": J,
    #     "E_to_GHz": E_to_GHz,
    #     "h": mu * H,
    #     "Nr": 4,
    #     "Ng": 4,
    # }
    # kwargs["N"] = 10
    # energies, ev, kvalues = get_dispersion_theta(0,
    #                                              10,
    #                                              return_eigenfunctions=True,
    #                                              parallel=False,
    #                                              **kwargs)
    with Data('test.h5') as f:
        # del kwargs["E_to_GHz"]
        # del kwargs["h"]
        # kwargs["H"] = H
        # f.save_data(kvalues, energies, ev, **kwargs)
        kvalues, energies, ev, attrs = f.read_data(1)
        print(kvalues)
