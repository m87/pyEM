from scipy.stats import multivariate_normal
import numpy as np
from utils import Model


class Stepwise(object):
    def __init__(self, n, arg):
        self.arg = arg
        self.n = n

    def __e(self, instance):
        for c in range(self.n):
            self.resps[c] = self.model.weights[c] * multivariate_normal.pdf(instance,self.model.means[c][0], self.model.covars[c])
        np.clip(self.resps, 0.000000001, np.inf, out=self.resps)
        self.resps /= sum(self.resps)

    def __m(self, instance):
        self.N += 1
        lam = np.power(self.N+2, -self.arg)
        self.histNk += self.resps
        self.histMean += self.resps*instance
        
        self.model.weights = self.model.weights * (1.0-lam) + lam * self.histNk / self.N
        self.model.weights /= sum(self.model.weights)
        
        self.model.means = (1.0 - lam) * self.model.means + lam * self.histMean / self.histNk


        for c in range(self.n):
            diff = np.array([instance]) - self.model.means[c]
            self.histCovar[c] +=  np.dot(self.resps[c] *diff.T, diff)
            #self.model.covars[c] = (1.0-lam) * self.model.covars[c] + lam * self.histCovar[c] / self.histNk[c]

    def __err(self, instance):
        pass


    def fit(self,X, n=10):
        self.dim = len(X[0])
        self.resps = np.zeros((self.n))
        self.histMean = np.zeros((self.n,1,self.dim))
        self.histCovar = np.zeros((self.n,self.dim,self.dim))
        self.histNk = np.zeros((self.n))
        self.model = Model(self.n, self.dim)
        self.N = 0
        for i in range(self.n):
            self.model.set(i, 1.0/n,(X[i],),np.eye(self.dim))

        for it, x in X:
            self.__e(np.array(x))
            self.__m(np.array(x))

        return self.model




class Entropy(object):
    def __init__(self, n, arg):
        self.arg = arg
        self.n = n

    def __e(self, instance):
        for c in range(self.n):
            self.resps[c] = self.model.weights[c] * multivariate_normal.pdf(instance,self.model.means[c][0], self.model.covars[c])
        np.clip(self.resps, 0.000000001, np.inf, out=self.resps)
        self.resps /= sum(self.resps)

    def __m(self, instance):
        self.N += 1
        lam = np.power(self.N+2, -self.arg)
        self.histNk += self.resps
        self.histMean += self.resps*instance
        
        self.model.weights = self.model.weights * (1.0-lam) + lam * self.histNk / self.N
        self.model.weights /= sum(self.model.weights)
        
        self.model.means = (1.0 - lam) * self.model.means + lam * self.histMean / self.histNk


        for c in range(self.n):
            diff = np.array([instance]) - self.model.means[c]
            self.histCovar[c] +=  np.dot(self.resps[c] *diff.T, diff)
            #self.model.covars[c] = (1.0-lam) * self.model.covars[c] + lam * self.histCovar[c] / self.histNk[c]

    def __err(self, instance):
        pass


    def fit(self,X, n=10):
        self.dim = len(X[0])
        self.resps = np.zeros((self.n))
        self.histMean = np.zeros((self.n,1,self.dim))
        self.histCovar = np.zeros((self.n,self.dim,self.dim))
        self.histNk = np.zeros((self.n))
        self.model = Model(self.n, self.dim)
        self.N = 0
        for i in range(self.n):
            self.model.set(i, 1.0/n,(X[i],),np.eye(self.dim))

        for it, x in X:
            self.__e(np.array(x))
            self.__m(np.array(x))

        return self.model


class IncrementalOne(object):
    def __init__(self, n, arg):
        self.arg = arg
        self.n = n

    def __e(self, instance):
        for c in range(self.n):
            self.resps[c] = self.model.weights[c] * multivariate_normal.pdf(instance,self.model.means[c][0], self.model.covars[c])
        np.clip(self.resps, 0.000000001, np.inf, out=self.resps)
        self.resps /= sum(self.resps)

    def __m(self, instance):
        self.N += 1
        lam = np.power(self.N+2, -self.arg)
        self.histNk += self.resps
        self.histMean += self.resps*instance
        
        self.model.weights = self.model.weights * (1.0-lam) + lam * self.histNk / self.N
        self.model.weights /= sum(self.model.weights)
        
        self.model.means = (1.0 - lam) * self.model.means + lam * self.histMean / self.histNk


        for c in range(self.n):
            diff = np.array([instance]) - self.model.means[c]
            self.histCovar[c] +=  np.dot(self.resps[c] *diff.T, diff)
            #self.model.covars[c] = (1.0-lam) * self.model.covars[c] + lam * self.histCovar[c] / self.histNk[c]

    def __err(self, instance):
        pass


    def fit(self,X, n=10):
        self.dim = len(X[0])
        self.resps = np.zeros((self.n))
        self.histMean = np.zeros((self.n,1,self.dim))
        self.histCovar = np.zeros((self.n,self.dim,self.dim))
        self.histNk = np.zeros((self.n))
        self.model = Model(self.n, self.dim)
        self.N = 0
        for i in range(self.n):
            self.model.set(i, 1.0/n,(X[i],),np.eye(self.dim))

        for it, x in X:
            self.__e(np.array(x))
            self.__m(np.array(x))

        return self.model

class IncrementalK(object):
    def __init__(self, n, arg):
        self.arg = arg
        self.n = n

    def __e(self, instance):
        for c in range(self.n):
            self.resps[c] = self.model.weights[c] * multivariate_normal.pdf(instance,self.model.means[c][0], self.model.covars[c])
        np.clip(self.resps, 0.000000001, np.inf, out=self.resps)
        self.resps /= sum(self.resps)

    def __m(self, instance):
        self.N += 1
        lam = np.power(self.N+2, -self.arg)
        self.histNk += self.resps
        self.histMean += self.resps*instance
        
        self.model.weights = self.model.weights * (1.0-lam) + lam * self.histNk / self.N
        self.model.weights /= sum(self.model.weights)
        
        self.model.means = (1.0 - lam) * self.model.means + lam * self.histMean / self.histNk


        for c in range(self.n):
            diff = np.array([instance]) - self.model.means[c]
            self.histCovar[c] +=  np.dot(self.resps[c] *diff.T, diff)
            #self.model.covars[c] = (1.0-lam) * self.model.covars[c] + lam * self.histCovar[c] / self.histNk[c]

    def __err(self, instance):
        pass


    def fit(self,X, n=10):
        self.dim = len(X[0])
        self.resps = np.zeros((self.n))
        self.histMean = np.zeros((self.n,1,self.dim))
        self.histCovar = np.zeros((self.n,self.dim,self.dim))
        self.histNk = np.zeros((self.n))
        self.model = Model(self.n, self.dim)
        self.N = 0
        for i in range(self.n):
            self.model.set(i, 1.0/n,(X[i],),np.eye(self.dim))

        for it, x in X:
            self.__e(np.array(x))
            self.__m(np.array(x))

        return self.model

class IncrementalInf(object):
    def __init__(self, n, arg):
        self.arg = arg
        self.n = n

    def __e(self, instance):
        for c in range(self.n):
            self.resps[c] = self.model.weights[c] * multivariate_normal.pdf(instance,self.model.means[c][0], self.model.covars[c])
        np.clip(self.resps, 0.000000001, np.inf, out=self.resps)
        self.resps /= sum(self.resps)

    def __m(self, instance):
        self.N += 1
        lam = np.power(self.N+2, -self.arg)
        self.histNk += self.resps
        self.histMean += self.resps*instance
        
        self.model.weights = self.model.weights * (1.0-lam) + lam * self.histNk / self.N
        self.model.weights /= sum(self.model.weights)
        
        self.model.means = (1.0 - lam) * self.model.means + lam * self.histMean / self.histNk


        for c in range(self.n):
            diff = np.array([instance]) - self.model.means[c]
            self.histCovar[c] +=  np.dot(self.resps[c] *diff.T, diff)
            #self.model.covars[c] = (1.0-lam) * self.model.covars[c] + lam * self.histCovar[c] / self.histNk[c]

    def __err(self, instance):
        pass


    def fit(self,X, n=10):
        self.dim = len(X[0])
        self.resps = np.zeros((self.n))
        self.histMean = np.zeros((self.n,1,self.dim))
        self.histCovar = np.zeros((self.n,self.dim,self.dim))
        self.histNk = np.zeros((self.n))
        self.model = Model(self.n, self.dim)
        self.N = 0
        for i in range(self.n):
            self.model.set(i, 1.0/n,(X[i],),np.eye(self.dim))

        for it, x in X:
            self.__e(np.array(x))
            self.__m(np.array(x))

        return self.model

