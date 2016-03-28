from scipy.stats import multivariate_normal
import numpy as np
from utils import Model

class Batch():
    """docstring for Batch"""
    def __init__(self, n, param, history=True, *args, **kwargs):
        self.n = n
        self.eps = -0.001


    def __e(self, instances):
        """E step of Batch EM.

        :instances: 1-D array
        :returns: None

        """
        for it, instance in instances:
            for c in range(self.n):
                self.ric[it][c] = self.model.weights[c] * multivariate_normal.pdf(instance, self.model.means[c][0],
                                                                          self.model.covars[c])

            np.clip(self.ric[it], 0.00000001, np.inf, out=self.ric[it])
            self.ric[it] /= sum(self.ric[it])



    def __m(self, instances):
        """M step of Batch EM

        :instances: 1-D array
        :returns: None

        """
        for c in range(self.n):
            nk = 0.0
            nkMean = 0.0
            for it, instance in instances:
                nk += self.ric[it][c]
                nkMean += self.ric[it][c] * instance

            self.model.means[c] = nkMean / nk
            self.model.weights[c] = nk / len(instances)

            nkCov = np.zeros((self.dim, self.dim))
            for it, instance in instances:
                diff = np.array([instance]) - self.model.means[c]
                mul = self.ric[it][c] * diff.T * diff
                nkCov += mul
            self.model.covars[c] = nkCov / nk

    def __err(self, instances):
        """TODO: Docstring for __err.

        :instances: 1-D array
        :returns: error

        """
        out = 0.0
        for it, instance in instances:
            s = 0.0
            for c in range(self.model.n):
                s += self.model.weights[c] * multivariate_normal.pdf(instance, self.model.means[c][0], self.model.covars[c])
            out += np.log(s)
        return out


    def fit(self, X, n=10):
        """TODO: Docstring for fit.

        :X: TODO
        :n: TODO
        :returns: TODO

        """
        self.dim = len(X[0])
        self.ric = np.zeros((len(X),self.n))
        self.model = Model(self.n, self.dim)
        self.N = 0
        for i in range(self.n):
            self.model.set(i, 1.0/n,(X[i],),np.eye(self.dim))



        err = -np.inf
        while True:
            self.__e(X)
            self.__m(X)

            newErr = self.__err(X)
            if err - newErr > self.eps:
                break
            print(err, "->", newErr)
            err = newErr

        return self.model





