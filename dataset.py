import numpy as np

class Dataset(object):

    def __init__(self, src, n, size=1):
        self.src = src
        self.n = n
        self.__it = -size
        self.size = size
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




