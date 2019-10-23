import numpy as np
import os
import yaml
from magnons.magnetization import magnetization_angle
from magnons.energies import get_dispersion_theta
from magnons.yig import a, S, mu, J, Ms
from magnons.cgs import E_to_GHz
from magnons.data import Data


class Process:
    config_name = 'config.yaml'

    def __init__(self, directory):
        files = os.listdir(directory)

        if self.config_name not in files:
            raise FileNotFoundError(
                "{config_name} does not exist in f{directory}")
        with open(os.path.join(directory, self.config_name), 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        if 'magnons' not in cfg:
            raise ValueError("Config file should contain a 'magnons' key")

        self.dir = cfg['magnons']['dir']
        self.save = cfg['magnons']['save']
        if 'runs' not in cfg:
            raise ValueError("Config file should contain at least one run")
        self.runs = cfg['runs']

    def run_all(self):
        for r in self.runs:
            # these are the default parameters for YIG
            kwargs = {
                "theta": 0,
                "Nk": 12,
                "alpha": 0,
                "phi": 0,
                "use_angled_if_zero": False,
                "eps": a**-2,
                "a": a,
                "S": S,
                "mu": mu,
                "J": J,
                "E_to_GHz": E_to_GHz,
                "H": 900,
                "Nr": 4,
                "Ng": 4,
                "N": 100,
                "Ms": Ms,
                "return_eigenfunctions": True,
                "ky_begin": 2,
                "ky_end": 6,
                "logspace": True,
                "parallel": True
            }
            list_of_kwargs = [
                "theta", "Nk", "phi", "alpha", "use_angled_if_zero",
                "ky_begin", "ky_end", "logspace", "parallel",
                "return_eigenfunctions", "N", "J", "S", "h", "eps", "a", "mu",
                "Nr", "Ng", "N"
            ]
            for k in r:
                if k in ['alpha', 'theta', 'phi']:
                    kwargs[k] = np.radians(r[k])
                else:
                    kwargs[k] = r[k]

            if 'alpha' in r and 'phi' not in r:
                kwargs['phi'] = magnetization_angle(kwargs['alpha'],
                                                    M=kwargs["Ms"],
                                                    H=kwargs["H"])
            kwargs['h'] = kwargs['H'] * mu
            apply_kwargs = {}
            for n in list_of_kwargs:
                if n in kwargs:
                    apply_kwargs[n] = kwargs[n]
                else:
                    print(f"{n} is missing in kwargs list")

            save_kwargs = {n: kwargs[n] for n in Data.supported_kwargs}
            for n in save_kwargs:
                if n in ['alpha', 'theta', 'phi']:
                    save_kwargs[n] = np.degrees(save_kwargs[n])

            with Data(self.save) as f:
                if f.find_if_exist(save_kwargs):
                    continue
            E, ev, kvalues = get_dispersion_theta(**apply_kwargs)

            with Data(self.save) as f:
                f.save_data(kvalues, E, ev, save_kwargs)
