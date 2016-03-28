import numpy as np

class Model():
    def __init__(self,n,dim):
        self.n = n
        self.dim = dim
        self.weights = np.zeros((n))
        self.means = np.zeros((n,1,dim))
        self.covars = np.zeros((n,dim,dim))

    def set(self, n , weight, mean, covar):
        self.weights[n] = weight
        self.means[n] = np.array(mean)
        self.covars[n] = np.array(covar)


class MultivariateGaussian(object):
    """model = collection"""

    def __init__(self, model):
        self.model = model

    def __getitem__(self, item):
        if self.model.n == 1:
            return np.random.multivariate_normal(self.model.means[0][0], self.model.covars[0])
        else:
            tmp = np.random.choice(range(self.model.n))
            return np.random.multivariate_normal(self.model.means[tmp][0], self.model.covars[tmp])

    def __len__(self):
        return np.inf

    def array(self, size):
        if self.model.n == 1:
            return np.random.multivariate_normal(self.model.means[0][0], self.model.covars[0], size)
        else:
            tmp = size // self.model.n
            array = []
            for model in range(self.model.n):
                array.extend(np.random.multivariate_normal(self.model.means[model][0], self.model.covars[model], tmp))
            np.random.shuffle(array)
            return array



class DataStream(object):
    """DataStream"""

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
            return self.__it, np.array(self.src[self.__it])
        else:
            return self.__it, np.array(self.src[self.__it:self.__it + self.size])

    def __next__(self):
        return self.next()

    def __iter__(self):
        self.__it = -self.size
        return self

    def __len__(self):
        return self.n * self.size

    def __getitem__(self, item):
        return np.array(self.src[item])
