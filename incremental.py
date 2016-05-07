import utils as uts
from thirdparty import log_mvnpdf, log_mvnpdf_diag
import numpy as np
from scipy.misc import logsumexp
import online
from gaussEM import GaussEM
EPS = np.finfo(float).eps


class Incremental(online.OnlineEM):
    def __init__(self, param):
        super().__init__(param)
        self.skip = int(param['skip'])

    def prepare(self, dataset):
        super().prepare(dataset)


class IncrementalGauss(Incremental, GaussEM):
    def __init__(self, param):
        super().__init__(param)
        self.cov = param['cov']
        self.k = param['k']
        self.select = param['select']
        self.C = float(param['smoothing'])
        
        self.histAcc = 0.0
        self.func = {
            'inf': (self.__e_inf, self.__m_inf),
            'one': (self.__e_inf, self.__m_one),
            'k': (self.__e_inf, self.__m_k),
        }
        self.mvnpdf = {'full': log_mvnpdf, 'diag': log_mvnpdf_diag}

    def __e_inf(self, X):
        lg = self.mvnpdf[self.cov](np.array([X]), self.means, self.COV[self.cov])
        logResps = lg[0] + np.log(self.weights)
        self.histAcc += logsumexp(logResps)
        self.hist.append(-self.histAcc/self.N)

        maxLg = np.max(logResps)
        logResps -= maxLg
        self.resps = np.exp(logResps)
        np.clip(self.resps, 10*EPS, np.inf, out=self.resps)
        self.resps /= np.sum(self.resps)

        
    def __m_inf(self,X):
        self.N += 1
        for c in range(self.n):
            self.accResps[c] += self.resps[c]
            self.accMeans[c] += X * self.resps[c]
            diff = X - (self.accMeans[c] + self.mu0[c]*self.C )  / (self.accResps[c]  + self.C)
            self.accCovars[c] +=  np.outer(self.resps[c] * diff, diff)

            self.means[c] = (self.accMeans[c] + self.mu0[c]*self.C )  / (self.accResps[c]  + self.C)
            self.weights[c] = (self.accResps[c] +self.C) /( self.N+self.n)
            self.covars[c] = (self.accCovars[c] + self.C *np.identity(self.dim) )/ (self.accResps[c]+self.C) * self.I[self.cov]
            self.diagCovars[c] = np.diag(self.covars[c])


    def __m_one(self,X):
        c=np.random.choice(range(self.n), p = self.resps)
        self.N += 1
        self.accResps[c] += 1.0
        self.accMeans[c] += X 
        diff = X - (self.accMeans[c] + self.mu0[c]*self.C )  / (self.accResps[c]  + self.C)
        self.accCovars[c] +=  np.outer(diff, diff)

        self.means[c] = (self.accMeans[c] + self.mu0[c]*self.C )  / (self.accResps[c]  + self.C)
        self.weights[c] = (self.accResps[c] +self.C) /( self.N+self.n)
        self.covars[c] = (self.accCovars[c] + self.C *np.identity(self.dim) )/ (self.accResps[c]+self.C)* self.I[self.cov]
        self.diagCovars[c] = np.diag(self.covars[c])


    def __m_k(self,X):
        self.N += 1
        self.ck=np.zeros((self.n,))
        for i in range(self.k):
            c=np.random.choice(range(self.n), p = self.resps)
            self.ck[c]+=1
        self.ck /= float(self.k)
        for c in range(self.n):
            self.accResps[c] += self.ck[c]
            self.accMeans[c] += X * self.ck[c]
            diff = X - (self.accMeans[c] + self.mu0[c]*self.C )  / (self.accResps[c]  + self.C)
            self.accCovars[c] +=  np.outer(self.ck[c] * diff, diff)

            self.means[c] = (self.accMeans[c] + self.mu0[c]*self.C )  / (self.accResps[c]  + self.C)
            self.weights[c] = (self.accResps[c] +self.C) /( self.N+self.n)
            self.covars[c] = (self.accCovars[c] + self.C *np.identity(self.dim) )/ (self.accResps[c]+self.C)* self.I[self.cov]
            self.diagCovars[c] = np.diag(self.covars[c])



    def e(self,X):
        self.func[self.select][0](X)

    def m(self,X):
        self.func[self.select][1](X)


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
        self.covars = np.array([np.identity(self.dim) for x in range(self.n)])
        self.diagCovars = np.ones((self.n,self.dim))
        self.COV = {'full' : self.covars, 'diag' : self.diagCovars}
        self.I ={'full': 1.0, 'diag': np.identity(self.dim)}

