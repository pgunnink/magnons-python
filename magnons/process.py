import numpy as np
import os
import yaml
from magnons.magnetization import magnetization_angle
from magnons.energies import get_dispersion_theta
from magnons.yig import a, S, mu, J, Ms
from magnons.cgs import E_to_GHz
from magnons.data import Data
from copy import copy
from itertools import product
import re


class Process:
    config_name = 'config.yaml'
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

    iterable_kwargs = ['alpha', 'N', 'theta']

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

    def print_all_data(self):
        with Data(self.save) as f:
            f.print_all_data()

    def process_run_for_iterable(self, r):
        res = []
        kwargs_that_are_list = []
        for it_kwarg in self.iterable_kwargs:
            if it_kwarg in r:
                if isinstance(r[it_kwarg], list):
                    kwargs_that_are_list.append(it_kwarg)
                elif isinstance(r[it_kwarg], str):
                    re_match = re.match(r".*:.*:.*", r[it_kwarg])
                    if re_match is not None:
                        r[it_kwarg] = np.arange(
                            *[float(x) for x in r[it_kwarg].split(':')])
                        kwargs_that_are_list.append(it_kwarg)
        if len(kwargs_that_are_list) == 0:
            res.append(r)
            return res
        else:
            for x in product(*[r[y] for y in kwargs_that_are_list]):
                r_copy = r.copy()
                for kwarg, value in zip(kwargs_that_are_list, x):
                    r_copy[kwarg] = value
                res.append(r_copy)
            return res

    def get_all_kwargs(self):
        r_new = []
        for r in self.runs:
            if 'runs' in r:
                for t in r['runs']:
                    r_copy = r.copy()
                    del r_copy['runs']
                    for key in t:
                        r_copy[key] = t[key]
                    r_new.extend(self.process_run_for_iterable(r_copy))
            else:
                r = self.process_run_for_iterable(r)
                r_new.extend(r)
        return r_new

    def get_all(self):
        for r in self.get_all_kwargs():
            apply_kwargs, save_kwargs = self.create_save_and_run_kwargs(r)
            with Data(self.save) as f:
                if f.find_if_exist(save_kwargs)[0]:
                    yield f.get_data(save_kwargs)
                else:
                    print("Calculating:")
                    print(r)
                    E, ev, kvalues = get_dispersion_theta(**apply_kwargs)
                    f.save_data(kvalues, E, ev, save_kwargs)
                    yield f.get_data(save_kwargs)

    def create_save_and_run_kwargs(self, run_kwargs):
        kwargs = copy(self.kwargs)
        list_of_kwargs = [
            "theta", "Nk", "phi", "alpha", "use_angled_if_zero", "ky_begin",
            "ky_end", "logspace", "parallel", "return_eigenfunctions", "N",
            "J", "S", "h", "eps", "a", "mu", "Nr", "Ng", "N", "Nk"
        ]
        for k in run_kwargs:
            if k in ['alpha', 'theta', 'phi']:
                kwargs[k] = np.radians(run_kwargs[k])
            else:
                kwargs[k] = run_kwargs[k]

        if 'alpha' in run_kwargs and 'phi' not in run_kwargs:
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

        return apply_kwargs, save_kwargs

    def run_all(self):
        for r in self.get_all_kwargs():
            apply_kwargs, save_kwargs = self.create_save_and_run_kwargs(r)
            with Data(self.save) as f:
                if f.find_if_exist(save_kwargs)[0]:
                    continue
            E, ev, kvalues = get_dispersion_theta(**apply_kwargs)

            with Data(self.save) as f:
                f.save_data(kvalues, E, ev, save_kwargs)
