import utils as uts
import numpy as np
import scipy
import online
from scipy.misc import logsumexp

EPS = np.finfo(float).eps
class Entropy(online.OnlineEM):
    def __init__(self, n_clusters, parma=None):
        super().__init__(n_clusters)
        self.lam= 0.9
        self.histAcc = 0.0

    def e(self, X):
        lg = uts.log_mvnpdf(np.array([X]), self.means, self.covars)
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
        tmp =(accWeights.sum() + 10 * EPS) + 10 * EPS
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
#            print(np.linalg.eigvals(ctmp))


