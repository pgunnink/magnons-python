from magnons.dipolar_sums import Dkxx, Dkzz, Dkxy
import numpy as np
import matplotlib.pyplot as plt
from magnons.yig import a, S, mu, J

if __name__ == '__main__':
    eps = a**(-2)
    for NN in [1, 2, 3, 4]:
        xx = Dkzz(eps=eps, a=a, mu=mu, Nr=NN, Ng=NN)
        plt.plot(np.real(xx.table(10**5, 10**5, 400)), label=NN)
    plt.legend()
    plt.show()
