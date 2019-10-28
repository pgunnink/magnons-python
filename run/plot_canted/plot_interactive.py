from magnons.interactive import DoublePlot
import matplotlib.pyplot as plt
import os
from magnons.process import Process

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))

    p = Process(dir_path)
    values = [[kvalues, E, ev, attrs] for kvalues, E, ev, attrs in p.get_all()]
    i = 3
    db = DoublePlot(values[i][0], values[i][1], values[i][2])
    db.plot_E(Nlim=50, logplot=False, ylim=(2, 15))
    print(values[i][3])
    plt.show()