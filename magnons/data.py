import h5py
import numpy as np
import os


class Data:
    supported_kwargs = [
        'eps', 'a', 'S', 'mu', 'J', 'H', 'Nr', 'Ng', 'phi', 'alpha', 'theta',
        'N', 'E_to_GHz', 'Nr', 'Ng', 'ky_begin', 'ky_end', 'logspace'
    ]
    data_path = 'data'
    counter_name = 'DATACOUNTER'

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        if os.path.isfile(self.path):
            self.file = h5py.File(self.path, 'r+')
        else:
            self.file = h5py.File(self.path, 'w')

        if self.data_path not in self.file.keys():
            self.file.create_group(self.data_path)
        self.data_path = self.file[self.data_path]
        if self.counter_name not in self.data_path.attrs:
            self.data_path.attrs.create(self.counter_name, 0)
        self.counter = self.data_path.attrs[self.counter_name]
        return self

    def __exit__(self, *args):
        self.file.close()

    def print_all_data(self):
        for r in self.data_path:
            print(dict(self.data_path[r].attrs))

    def find_if_exist(self, kwargs):
        for r in self.data_path:
            attrs = dict(self.data_path[r].attrs)
            if len(attrs) != len(kwargs):
                continue
            overlap = True
            for n in attrs:
                if n not in kwargs:
                    overlap = False
                    break
                value_attrs = attrs[n]
                value_kwargs = kwargs[n]
                # first check if both are 0
                if value_attrs == 0 and value_kwargs == 0:
                    continue
                # if either one is 0, but the other is not, break
                if value_attrs == 0 or value_kwargs == 0:
                    overlap = False
                    break
                if isinstance(value_attrs, np.bool_) and isinstance(
                        value_kwargs, bool):
                    if value_attrs != value_kwargs:
                        overlap = False
                        break

                if np.abs(1 - value_attrs / value_kwargs) > 0.02:
                    overlap = False
                    break
            if overlap:
                return True, r
        return False, None

    def get_data(self, kwargs):
        exist, r = self.find_if_exist(kwargs)
        if not exist:
            raise Exception("One of data files does not exist")
        path = self.data_path[r]
        energies = path['energies']
        ev = path['ev']
        kvalues = path['kvalues']
        attrs = dict(path.attrs)
        return np.array(kvalues), np.array(energies), np.array(ev), attrs

    def save_data(self, kvalues, energies, ev, kwargs, check_exist=True):
        for k in kwargs:
            if k not in self.supported_kwargs:
                raise Exception(f"{k} is not a supported keyword to save")
        name = f"{self.counter+1}"
        if check_exist and self.find_if_exist(kwargs)[0]:
            return "Already exists"

        try:
            path = self.data_path.create_group(name)
            path.create_dataset('energies', data=energies)
            path.create_dataset('ev', data=ev)
            path.create_dataset('kvalues', data=kvalues)
            for k in kwargs:
                path.attrs[k] = kwargs[k]
        except:
            raise Exception("Something went wrong saving datafile")
        self.data_path.attrs.modify(self.counter_name, self.counter + 1)

    def read_data(self, i):
        if f"{i}" not in self.data_path:
            raise Exception("Not in path")
        path = self.data_path[f"{i}"]
        energies = path['energies']
        ev = path['ev']
        kvalues = path['kvalues']
        attrs = dict(path.attrs)
        return np.array(kvalues), np.array(energies), np.array(ev), attrs