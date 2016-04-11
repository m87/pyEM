import utils as uts
import numpy as np

class Batch(object):
    def __init__(self, n_clusters, parma=None):
        self.n = n_clusters
        self.param = int(parma)

    def __e(self, X):
        self.resps = self.weights * np.exp(uts.log_mvnpdf(np.array(X[:self.param]), self.means, self.covars))
        np.clip(self.resps, 0.000000000001, np.inf, out=self.resps)
        self.resps /= np.sum(self.resps, axis=1)[:,None]

    def __m(self, X):
        self.sumResps = np.sum(self.resps, axis=0)
        self.sumMeans = np.sum((self.resps.T[:,None]*X[:self.param].T), axis=2)
        self.weights = self.sumResps / self.param
        self.means =  self.sumMeans / self.sumResps[:,None]



    def fit(self, dataset):
        #print(np.exp(uts.log_mvnpdf(np.array([[1,1]]), np.array([[1,1]]), np.array([[[1,0],[0,1]]]))))
        #print(dataset.shape())
        self.__prepare(dataset)
        for i in range(10):
            print(i)
            self.__e(dataset)
            self.__m(dataset)



    def __prepare(self, dataset):
        shape = dataset.shape()
        dim = shape[0][0]
        self.N = 0;
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


