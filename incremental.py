import utils as uts
import numpy as np
from scipy.misc import logsumexp
import online
EPS = np.finfo(float).eps

class Incremental(online.OnlineEM):
    def __init__(self, n_clusters, param, k=50):
        super().__init__(n_clusters)
        self.k = k
        self.select = param[0]
        self.lam = float(param[1])
        self.histAcc = 0.0
        self.func = {
            'inf': (self.__e_inf, self.__m_inf),
            'one': (self.__e_inf, self.__m_one),
        }

    def __e_inf(self, X):
        lg = uts.log_mvnpdf(np.array([X]), self.means, self.covars)
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
        #for c in range(self.n):
        #    tmpR = self.resps[c]
        #    tmpM = X * self.resps[c]
        #    self.weights[c] = self.weights[c] +tmpR - self.sfR[self.I]
        #    self.means[c] = self.means[c] + tmpM - self.sfM[self.I]
        #    self.sfR[self.I] = tmpR
        #    self.sfM[self.I] = tmpM
        #self.I += 1

        #print(self.weights)
        for c in range(self.n):
            self.accResps[c] += self.resps[c]
            self.accMeans[c] += X * self.resps[c]
            diff = X - self.accMeans[c] / self.accResps[c]
            self.accCovars[c] +=  np.outer(self.resps[c] * diff, diff)

            if self.N < 30 : return
            self.means[c] = self.accMeans[c] / self.accResps[c]
            self.weights[c] = self.accResps[c] / self.N
            self.covars[c] = self.accCovars[c] / self.accResps[c]


    def __m_one(self,X):
        c=np.random.choice(range(self.n), p = self.resps)
        self.N += 1
        self.accResps[c] += self.resps[c]
        self.accMeans[c] += X * self.resps[c]
        diff = X - self.accMeans[c] / self.accResps[c]
        self.accCovars[c] +=  np.outer(self.resps[c] * diff, diff)

        if self.N < 30: return
        self.means[c] = self.accMeans[c] / self.accResps[c]
        if self.resps[c] > 0.0000001:
            self.weights[c] = self.accResps[c] / self.N

        self.covars[c] = self.accCovars[c] / self.accResps[c]





    def e(self,X):
        self.func[self.select][0](X)

    def m(self,X):
        self.func[self.select][1](X)


    def prepare(self, dataset):
        super().prepare(dataset)
        self.suffResps = np.random.random((self.n, 20))
        self.weights= np.sum(self.suffResps, axis=1)
        self.suffMeans = np.random.random((self.n,))
        for i in range(self.n):
            self.suffMeans[i] = np.sum(np.random.random((self.dim,20))) 
        self.accRespsOld = np.random.random((self.n,))
        self.accMeansOld = np.random.random((self.n,self.dim))
        self.accCovarsOld = np.zeros((self.n,self.dim,self.dim))

        self.accMeans = self.means*10
        self.accResps = np.ones((self.n,)) * 10
 #       self.covars = np.array([np.identity(self.dim)*1 for x in range(self.n)])

        self.sfM = np.random.random((20, self.n, self.dim))
        self.sfR = np.random.random((20, self.n))
        #self.means = np.sum(self.sfM, axis=2)
        self.I = 0


