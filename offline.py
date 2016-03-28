from scipy.stats import multivariate_normal
import numpy as np
from utils import Model
from utils import Stats
from utils import vec

class Batch():
    def __init__(self, n, param, history=True, *args, **kwargs):
        self.n = n
        self.eps = -0.001


    def __e(self, instances):
        for it, instance in instances:
            for c in range(self.n):
                self.ric[it][c] = self.model.weights[c] * multivariate_normal.pdf(instance, self.model.means[c][0],
                                                                      self.model.covars[c])

            np.clip(self.ric[it], 0.00000001, np.inf, out=self.ric[it])
            self.ric[it] /= np.sum(self.ric[it])

    def __m(self, instances):
        for c in range(self.n):
            nk = 0.0
            nkMean = 0.0
            for it, instance in instances:
                nk += self.ric[it][c]
                nkMean += self.ric[it][c] * vec(instance)

            self.model.means[c] = nkMean / nk
            self.model.weights[c] = nk / len(instances)

            nkCov = np.zeros((self.dim, self.dim))
            for it, instance in instances:
                diff = vec(instance) - self.model.means[c]
                mul = self.ric[it][c] * diff.T * diff
                nkCov += mul
            self.model.covars[c] = nkCov / nk

    def __err(self, instances):
        out = 0.0
        for it, instance in instances:
            s = 0.0
            for c in range(self.model.n):
                s += self.model.weights[c] * multivariate_normal.pdf(instance, self.model.means[c][0], self.model.covars[c])
            out += np.log(s)
        self.stats.error.append(out)
        return out


    def fit(self, X, n=10):
        self.dim = len(X[0])
        self.ric = np.zeros((len(X),self.n))
        self.model = Model(self.n, self.dim)
        self.stats = Stats()
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



class BatchEntropy():
    def __init__(self, n, param, history=True, *args, **kwargs):
        self.n = n
        self.eps = -0.001
        self.arg = param


    def __e(self, instances):
        for it, instance in instances:
            for c in range(self.n):
                self.ric[it][c] = self.model.weights[c] * multivariate_normal.pdf(instance, self.model.means[c][0],
                                                                      self.model.covars[c])

            np.clip(self.ric[it], 0.00000001, np.inf, out=self.ric[it])
            self.ric[it] /= np.sum(self.ric[it])

    def __m(self, instances):
        sumWeights = np.zeros((self.n))
        sumMeans = np.zeros((self.n,1,self.dim))
        for c in range(self.n):
            sumWeights += self.ric[c]
            for it, instance in instances:
                sumMeans[c] += self.ric[it][c] * (vec(instance) - self.model.means[c])
        
        for c in range(self.n):
            sumWeights[c] *= -self.arg/float(len(instances))
            sumWeights[c] = np.exp(sumWeights[c]) * self.model.weights[c]
        sumAll = sum(sumWeights)

        for c in range(self.n):
            self.model.weights[c] = sumWeights[c] / sumAll
            self.model.means[c] += self.arg/float(len(instances)) * sumMeans[c]

        sumCovars = np.zeros((self.n,self.dim,self.dim))
        for c in range(self.n):
            iC = np.linalg.inv(self.model.covars[c])
            for it, instance in instances:
                diff = vec(instance) - self.model.means[c]
                sumCovars[c] += self.ric[it][c] * ( iC.dot(diff.T).dot(diff).dot(iC) ) 
            sumCovars[c] *= self.arg/float(len(instances)) 
            self.model.covars[c] = np.linalg.inv(iC + sumCovars[c])
            print(self.model.covars[c])


    def __err(self, instances):
        out = 0.0
        for it, instance in instances:
            s = 0.0
            for c in range(self.model.n):
                s += self.model.weights[c] * multivariate_normal.pdf(instance, self.model.means[c][0], self.model.covars[c])
            out += np.log(s)
        self.stats.error.append(out)
        return out


    def fit(self, X, n=10):
        self.dim = len(X[0])
        self.ric = np.zeros((len(X),self.n))
        self.model = Model(self.n, self.dim)
        self.stats = Stats()
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






