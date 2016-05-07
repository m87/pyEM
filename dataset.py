import numpy as np
from config import *
import utils
from loaders import *
from generators import fixed_generator

class Dataset(object):

    def __init__(self, src, labels=None, init=None, norm=None):
        if norm is None:
            self.X = np.array(src)
            self.I = np.array(init)
        else:
            self.X = np.array(src)/norm
            self.I = np.array(init)/norm

        self.L = np.array(labels)
        self.N = len(self.X)

    def randomize(self):
        t = np.array(zip(self.X, self.L))
        np.random.shuffle(t)
        t = np.array(zip(*t))
        self.X = np.array(t[0])
        self.L = np.array(t[1])

    def shape(self):
        instance = self.X[0];
        return np.shape(instance), self.N

    def __len__(self):
        return self.N

    def __getitem__(self, item):
        return self.X[item]



def get_dataset(config):
    if config.dataset_type == FIXED_GEN:
        model = config.dataset_params
        ar, ini, labels = fixed_generator(model, config.dataset_n, config.dataset_init)
        stream = Dataset(src=ar, init=ini, labels=labels)
        return stream

    if config.dataset_type == LIM_GEN:
        pass

    if config.dataset_type == MNIST:
        ar,ini,labels =  mnist_loader(config)
        stream = Dataset(src=ar, init=ini, norm=config.dataset_params[NORM], labels=labels)
        return stream

    if config.dataset_type == COVERTYPE:
        ar,ini,labels =  covertype_loader(config)
        stream = Dataset(src=ar, n=config.dataset_n, size=1, init=ini, c=config.alg_params[CLUSTERS],  labels=labels)
        return stream





