import h5py
import numpy as np
import os


class Data:
    supported_kwargs = [
        'eps', 'a', 'S', 'mu', 'J', 'H', 'Nr', 'Ng', 'phi', 'alpha', 'theta',
        'N', 'E_to_GHz', 'h'
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

    def save_data(self, kvalues, energies, ev, **kwargs):
        for k in kwargs:
            if k not in self.supported_kwargs:
                raise Exception(f"{k} is not a supported keyword to save")
        name = f"{self.counter+1}"
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