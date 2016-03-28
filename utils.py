from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2
from sklearn import mixture as mix
import numpy as np

class VecError(Exception): pass

def vec(src):
    if len(np.shape(src)) == 2 and np.shape(src)[0] == 1:
        return np.array(src)
    if len(np.shape(src)) == 1:
        return np.array([src])
    
    raise VecError()


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



def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def display_result(model, instances):
    plt.plot()
    plt.title('model')

    i = list(zip(*instances))
    plt.scatter(x=i[0], y=i[1])
    ax = None
    if ax is None:
        ax = plt.gca()

    for mo in range(model.n):
        vals, vecs = eigsorted(model.covars[mo])
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        width, height = 2 * np.sqrt(chi2.ppf(0.5, 2)) * np.sqrt(vals)
        ellip = Ellipse(xy=model.means[mo,0], width=width, height=height, angle=theta, alpha=0.5, color='red')

        ax.add_artist(ellip)

    plt.show()


def display_err(err):
    plt.plot(err)
    plt.show()




class Stats():
    def __init__(self):
        self.error = []

