import numpy as np
from config import *
import utils
from loaders import *
from generators import fixed_generator

class Dataset(object):

    def __init__(self, src, n, size=1, labels=None, init=None, c=None, norm=1.0):
        self.src = np.array(src)/norm
        self.labels = np.array(labels)
        self.n = n
        self.__it = -size
        self.size = size
        self.init = init

        if len(src) < n * size:
            raise IndexError

    def next(self):
        self.__it += self.size

        if self.__it >= self.n * (self.size):
            raise StopIteration

        if self.size == 1:
            return self.__it, self.src[self.__it]
        else:
            return self.__it, self.src[self.__it:self.__it + self.size]

    def getInit(self):
        return self.init

    def label(self):
        if self.size == 1:
            return self.labels[self.__it]
        else:
            return self.labels[self.__it:self.__it + self.size]

    def shape(self):
        instance = self.src[0];
        return np.shape(instance), self.n

    def __next__(self):
        return self.next()

    def __iter__(self):
        self.__it = -self.size
        return self

    def __len__(self):
        return self.n * self.size

    def __getitem__(self, item):
        return self.src[item]






def get_stream(config):
    if config.dataset_type == FIXED_GEN:
        model = config.dataset_params
        ar, ini, labels = fixed_generator(model, config.dataset_n, config.dataset_init)
        stream = Dataset(src=ar, n=config.dataset_n, size=1, init=ini, c=config.alg_params[CLUSTERS], labels=labels)
        return stream

    if config.dataset_type == LIM_GEN:
        pass

    if config.dataset_type == MNIST:
        ar,ini,labels =  mnist_loader(config)
        stream = Dataset(src=ar, n=config.dataset_n, size=1, init=ini, c=config.alg_params[CLUSTERS], norm=config.dataset_params[NORM], labels=labels)
        return stream

    if config.dataset_type == COVERTYPE:
        pass




