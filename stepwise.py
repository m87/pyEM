import utils as uts
import numpy as np

class Stepwise(object):
    def __init__(self, n_clusters, parma=None):
        self.n = n_clusters

    def __e(self, X):
        self.resps = self.weights * np.exp(uts.log_mvnpdf(np.array([X]), self.means, self.covars))
        np.clip(self.resps, 0.000000000001, np.inf, out=self.resps)
        self.resps = np.array(self.resps[0])
        self.resps /= self.resps.sum()

    def __m(self, X):
        self.N += 1
        lam = 1.0/self.N
        self.accResps += self.resps
        self.accMeans += X * self.resps[:,None]
        
        self.weights *= (1.0 - lam)
        self.weights += lam * self.accResps / self.N

        self.means *= (1.0 - lam)
        self.means += lam * self.accMeans / self.accResps[:,None]

        for c in np.arange(self.n):
            diff = X - self.means[c]
            self.accCovars[c] +=  np.outer(self.resps[c] * diff, diff)

        self.covars *= (1.0 - lam)
        self.covars += lam * self.accCovars / self.accResps[:,None,None]



    def fit(self, dataset):
        #print(np.exp(uts.log_mvnpdf(np.array([[1,1]]), np.array([[1,1]]), np.array([[[1,0],[0,1]]]))))
        #print(dataset.shape())
        self.__prepare(dataset)
        for it, X in dataset:
            print(it)
            self.__e(X)
            self.__m(X)



    def __prepare(self, dataset):
        shape = dataset.shape()
        dim = shape[0][0]
        self.N = 0;
        self.accResps = np.zeros((self.n,))
        self.accMeans = np.zeros((self.n,dim))
        self.accCovars = np.zeros((self.n,dim,dim))
        self.weights = np.ones((self.n,))
        self.weights /= self.n
        self.means = np.zeros((self.n,dim))
        for it in range(self.n):
            self.means[it] = dataset[it]
        self.covars = np.array([np.identity(dim) for x in range(self.n)])



    def __str__(self):
        out = ""
        np.set_printoptions(threshold=np.nan)
        out += 'w: ' + str(self.weights) + '\nm: ' + str(self.means)# + '\nc: ' + str(self.covars)
        return out


