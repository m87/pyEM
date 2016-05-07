import utils as uts
import numpy as np
import scipy
import online
from scipy.misc import logsumexp
from config import *
from gaussEM import GaussEM
from thirdparty import log_mvnpdf, log_mvnpdf_diag


class Entropy(online.OnlineEM):
    def __init__(self, param):
        super().__init__(param)
        self.skip = int(param['skip'])
        self.C  = float(param['smoothing'])
        self.lr= float(param['lr'])
        self.IT= int(param['iter'])
        self.alpha= float(param['alpha'])

    def prepare(self, dataset):
        super().prepare(dataset)


class EntropyGauss(Entropy, GaussEM):
    def __init__(self, param):
        super().__init__(param)
        self.histAcc = 0.0
        self.cov = param['cov']
        self.mvnpdf = {'full': log_mvnpdf, 'diag': log_mvnpdf_diag}

    def e(self, X):
        #print(self.means)
        self.N += 1
        lg = self.mvnpdf[self.cov](np.array([X]), self.means, self.COV[self.cov])
        logResps = lg[0] + np.log(self.weights)
        self.histAcc += logsumexp(logResps)
        self.hist.append(-self.histAcc/self.N)

        maxLg = np.max(logResps)
        logResps -= maxLg
        self.resps = np.exp(logResps)
        #self.resps /= np.sum(self.resps)
        self.resps /= np.sum(self.resps)

    def m(self, X):
        self.lam = np.power(self.N,-self.alpha)
        if self.N > self.IT:
            self.lam = self.lr
        accWeights = self.weights *np.exp( -self.lam * self.resps)
        tmp = accWeights.sum() #+ 10 * online.EPS) + 10 * online.EPS
        self.weights = accWeights / tmp
        #print(self.ICovars[0])
        #np.clip(self.ICovars, 0.00000000001, np.inf, self.ICovars)

        #lr = self.lam * self.resps

        for c in np.arange(self.n):

            self.means[c] += self.lam * self.resps[c] * (X - self.means[c])
            diff = X - self.means[c]


            self.ICovars[c] += self.lam * self.resps[c]*(self.ICovars[c] - np.dot(self.ICovars[c],diff[:,None]).dot(np.array([diff])).dot(self.ICovars[c]))
            #print(self.ICovars[c])
            self.covars[c] = scipy.linalg.inv(self.ICovars[c]) * self.I[self.cov]
            self.diagCovars[c] = np.diag(self.covars[c])
  #          self.means[c] += self.lam * self.resps[c] * (X - self.means[c])


    def prepare(self, dataset):
        super().prepare(dataset)
        self.accResps = np.zeros((self.n,)) 
        self.accMeans = np.zeros((self.n,self.dim))
        self.accCovars = np.zeros((self.n,self.dim,self.dim))
        self.weights = np.ones((self.n,))
        self.weights /= self.n
        self.means = np.zeros((self.n,self.dim))
        self.mu0 = np.zeros((self.n,self.dim))
        for it,x in enumerate(dataset.getInit()):
            self.means[it] =  x
            self.mu0[it] =  x
        print(self.means)
        self.covars = np.array([np.identity(self.dim) for x in range(self.n)])
        self.ICovars = np.array([scipy.linalg.inv(x) for x in self.covars]) 
        self.diagCovars = np.ones((self.n,self.dim)) 
        self.COV = {'full' : self.covars, 'diag' : self.diagCovars}
        self.I ={'full': 1.0, 'diag': np.identity(self.dim)}


