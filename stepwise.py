import utils as uts
import numpy as np
from online import *
from scipy.misc import logsumexp

class Stepwise(OnlineEM):
    def __init__(self, param):
        super().__init__(param)
        self.C = float(param['smoothing'])
        self.param = float(param['alpha'])
        self.skip = int(param['skip'])
        self.cov = param['cov']


    def predict(self, dataset):
        self.prepare(dataset)
        res = np.zeros((self.n,))
        for it, X in dataset:
            self.e(X)
            res[np.argmax(self.resps)] +=1

        return res

    def e(self, X):
        lg = mvnpdf[self.cov](np.array([X]), self.means, self.COV[self.cov])
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
        lam = 1.0/(np.power(self.N, float(self.param)))
        for c in np.arange(self.n):
            self.accResps[c]= (1-lam) * self.accResps[c] + lam * self.resps[c]
            self.accMeans[c]= (1-lam)* self.accMeans[c] + lam * X * self.resps[c]
            tmp =  self.accMeans[c] / self.accResps[c] 
            diff = X - tmp
            self.accCovars[c] = (1-lam) * self.accCovars[c] + lam *  np.outer(self.resps[c] * diff, diff) 


        self.accResps /= np.sum(self.accResps)

    def m(self, X):
        if self.N < self.skip: return
        lam = 1.0/(np.power(self.N, float(self.param)))
        for c in np.arange(self.n):
            self.weights[c] = self.accResps[c] / (self.N+ 10*EPS ) + EPS

            self.means[c] =  self.accMeans[c] / (self.accResps[c] + 10 * EPS )

            self.covars[c] = (self.accCovars[c] + 10* EPS * np.identity(self.dim))/ (self.accResps[c] + 10 * EPS ) * self.I[self.cov]
            self.diagCovars[c] = np.diag(self.covars[c])

        self.weights /= sum(self.weights)



