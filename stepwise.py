from thirdparty import log_mvnpdf, log_mvnpdf_diag
import numpy as np
from online import *
from scipy.misc import logsumexp
from gaussEM import GaussEM

class Stepwise(OnlineEM):
    def __init__(self, param):
        super().__init__(param)
        self.param = float(param['alpha'])
        self.skip = int(param['skip'])
        self.mbsize= int(param['mb'])

    def prepare(self, dataset):
        super().prepare(dataset)

class StepwiseGauss(Stepwise, GaussEM):
    def __init__(self, param):
        super().__init__(param)
        self.cov = param['cov']
        self.C = float(param['smoothing'])
        self.mvnpdf = {'full': log_mvnpdf, 'diag': log_mvnpdf_diag}


    def e(self, X):
        lg = self.mvnpdf[self.cov](np.array([X]), self.means, self.COV[self.cov])
        #s = np.inner((X - self.means),(X-self.means))
        #print(s)
        #print(self.means[0])
        logResps = lg[0] + np.log(self.weights)
        self.histAcc += logsumexp(logResps)
        self.hist.append(-self.histAcc/self.N)
        #self.hist.append(logsumexp(logResps))
        maxLg = np.max(logResps)
        logResps -= maxLg
        self.resps = np.exp(logResps)
        np.clip(self.resps, 10*EPS, np.inf, out=self.resps)
        
        self.resps /= np.sum(self.resps)
        self.N += 1
        lam = np.power(self.N+2, -float(self.param))
        for c in np.arange(self.n):
            self.accResps[c]= (1-lam) * self.accResps[c] + lam * self.resps[c]
            self.accMeans[c]= (1-lam)* self.accMeans[c] + lam * X * self.resps[c]
            tmp =  self.accMeans[c] / self.accResps[c] 
            diff = X - tmp
            self.accCovars[c] = (1-lam) * self.accCovars[c] + lam *  np.outer(self.resps[c] * diff, diff) 


        self.accResps /= np.sum(self.accResps)

    def m(self, X):
        if self.N < self.skip: return
        if self.N % self.mbsize != 0:
            return
        for c in np.arange(self.n):
            self.weights[c] = self.accResps[c] / (self.N+ 10*EPS ) + EPS
            self.means[c] = (self.accMeans[c] + 10* EPS)/ (self.accResps[c] + 10 * EPS ) 


            self.covars[c] = (self.accCovars[c] + 10* EPS * np.identity(self.dim))/ (self.accResps[c] + 10 * EPS ) * self.I[self.cov]
            self.diagCovars[c] = np.diag(self.covars[c])

        self.weights /= sum(self.weights)



    def prepare(self,dataset):
        super().prepare(dataset)
        self.accResps = np.zeros((self.n,))
        self.accMeans = np.zeros((self.n,self.dim))
        self.accCovars = np.zeros((self.n,self.dim,self.dim))
        self.weights = np.ones((self.n,))
        self.weights /= self.n
        self.means = np.zeros((self.n,self.dim))
        for it,x in enumerate(dataset.I):
            self.means[it] = x
        self.covars = np.array([np.identity(self.dim) for x in range(self.n)])
        self.diagCovars = np.ones((self.n,self.dim))
        self.COV = {'full' : self.covars, 'diag' : self.diagCovars}
        self.I ={'full': 1.0, 'diag': np.identity(self.dim)}



