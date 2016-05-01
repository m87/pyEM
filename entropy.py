import utils as uts
import numpy as np
import scipy
import online
from scipy.misc import logsumexp
from config import *


class Entropy(online.OnlineEM):
    def __init__(self, param):
        super().__init__(param[CLUSTERS])
        self.lam= 0.9
        self.histAcc = 0.0
        self.cov = param['cov']

    def e(self, X):
        print(self.covars)
        lg = online.mvnpdf[self.cov](np.array([X]), self.means, self.COV[self.cov])
        logResps = lg[0] + np.log(self.weights)
        self.histAcc += logsumexp(logResps)
        self.hist.append(-self.histAcc/self.N)

        maxLg = np.max(logResps)
        logResps -= maxLg
        self.resps = np.exp(logResps)
        self.resps /= np.sum(self.resps)
        self.N += 1

    def m(self, X):
        accWeights = self.weights *np.exp( -self.lam * self.resps)
        tmp =(accWeights.sum() + 10 * online.EPS) + 10 * online.EPS
        self.weights = accWeights / tmp
        self.lam -= 0.00001


        for c in np.arange(self.n):
            self.means[c] += self.lam * self.resps[c] * (X - self.means[c])

         #   self.means[c] += self.lam * self.resps[c]*diff

            self.weights /= sum(self.weights)
            diff = X - self.means[c]
            iC = scipy.linalg.inv(self.covars[c])
            d = np.array((diff,))
            iC = iC +self.lam * self.resps[c] * (iC - np.dot(iC,d.T).dot(d).dot(iC) ) #* np.identity(self.dim)
            ctmp = scipy.linalg.inv(iC) #+ 0.001 * np.ones((self.dim, self.dim))
            self.covars[c] = ctmp
            self.diagCovars[c] = np.diag(self.covars[c])
#            print(np.linalg.eigvals(ctmp))


