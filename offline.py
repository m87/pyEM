import utils as uts
import numpy as np
from scipy.misc import logsumexp
from config import *
from thirdparty import log_mvnpdf, log_mvnpdf_diag
from scipy.stats import multivariate_normal
EPS = np.finfo(float).eps

class BatchGauss(object):
    def __init__(self, param):
        self.n = param[CLUSTERS]
        self.th = float(param['th'])
        self.IT = int(param['iter'])
        self.covt = param['cov']
        self.param = int(param['n'])
        self.mvnpdf = {'full': log_mvnpdf, 'diag': log_mvnpdf_diag}
        self.hist = []

    def __e(self, X):
        lg = self.mvnpdf[self.covt](np.array(X[:self.param]), self.means, self.COV[self.covt])
        logResps = lg + np.log(self.weights)
        self.hist.append(-np.sum(logsumexp(logResps,axis=1))/self.N)
        maxLg = np.max(logResps)
        logResps -= maxLg
        self.resps = np.exp(logResps)
        np.clip(self.resps, 10*EPS, np.inf, out=self.resps)
        self.resps /= np.sum(self.resps, axis=1)[:, None]

    def __m(self, X):
        tmpResps = self.resps.sum(axis=0)
        respsMeanSum = np.dot(self.resps.T, X[:self.param])
        invRMS = 1.0 / (tmpResps[:, np.newaxis] + 10 * EPS)
        self.weights = (tmpResps / (np.sum(tmpResps) + 10 * EPS) + EPS)
        self.means = respsMeanSum * invRMS
        self.cov = np.zeros((self.n,self.dim, self.dim))
        for c in range(self.n):
            post = self.resps[:,c]
            diff = X[:self.param] - self.means[c]
            av = np.dot(post * diff.T, diff) / np.sum(post)
            self.covars[c] = av
            self.diagCovars[c] = np.diag(self.covars[c])

    def predict(self, X):
        lg = self.log_mvnpdf[self.covt](np.array([X]), self.means, self.COV[self.covt])
        logResps = lg + np.log(self.weights)
        maxLg = np.max(logResps)
        logResps -= maxLg
        self.resps = np.exp(logResps)
        self.resps /= np.sum(self.resps, axis=1)[:, None]
        return np.argmax(self.resps)
        
    def load(self, weights, means, covars):
        self.weights = np.load(weights)
        self.means = np.load(means)
        self.covars = np.load(covars)
        self.diagCovars = np.zeros((self.dim,))
        for c in self.covars:
            self.diagCovars[c] = np.diag(self.covars[c])

    def fit(self, dataset):
        self.__prepare(dataset)
        j=0
        for i in range(2):
            print(i)
            self.__e(dataset)
            self.__m(dataset)
        while True:
            print(j)
            j+=1
            self.__e(dataset)
            if abs(self.hist[-1] - self.hist[-2]) <= self.th:
                return
            if j > self.IT:
                return
            self.__m(dataset)



    def __prepare(self, dataset):
        shape = dataset.shape()
        self.dim = shape[0][0]
        self.N = len(dataset);
        self.weights = np.ones((self.n,))
        self.weights /= self.n
        self.means = np.zeros((self.n,self.dim))
        for it in range(self.n):
            self.means[it] = dataset[it]
        self.covars = np.array([np.identity(self.dim) for x in range(self.n)])
        self.diagCovars = np.ones((self.n,self.dim))
        self.COV = {'full' : self.covars, 'diag' : self.diagCovars}
        self.I ={'full': 1.0, 'diag': np.identity(self.dim)}



    def __str__(self):
        out = ""
        np.set_printoptions(threshold=np.nan)
        out += 'w: ' + str(self.weights) + '\nm: ' + str(self.means) + '\nc: ' + str(self.covars)
        return out
    
    def save(self, path):
        np.save(path+"/weights", self.weights)
        np.save(path+"/means", self.means)
        np.save(path+"/covars", self.covars)
        np.save(path+"/hist", self.hist)

class BatchEntropy(object):
    def __init__(self, n_clusters, parma=None):
        self.n = n_clusters
        self.param = int(parma)
        self.lam=0.9
        self.hist = []

    def __e(self, X):
        self.resps = self.weights * np.exp(uts.log_mvnpdf(np.array(X[:self.param]), self.means, self.covars))
        np.clip(self.resps, 0.00000000000000000001, np.inf, out=self.resps)
        self.resps /= np.sum(self.resps, axis=1)[:, None]

    def __m(self, X):
        self.sumResps = np.sum(self.resps, axis=0)
        self.weights  = self.weights * np.exp(np.dot(self.sumResps, -self.lam/self.param))
        self.weights /= np.sum(self.weights)


        self.sumMeans = np.zeros((self.n, self.dim))
        for c in range(self.n):
            for it,i in enumerate(X[:self.param]):
                diff = i - self.means[c]
                self.sumMeans[c] += np.dot(self.resps[it][c],diff)

            self.means[c] += np.dot(self.lam/self.param, self.sumMeans[c])

            iC =np.linalg.pinv(self.covars[c])
            nkCov = np.zeros((self.dim, self.dim))
            for it, instance in enumerate(X[:self.param]):
                diff = instance - self.means[c]
                nkCov += self.resps[it][c] * (iC - np.dot(np.dot(iC,diff[:,None])*diff,iC ))
            iC +=np.dot(self.lam/self.param , nkCov)
            self.covars[c] = np.linalg.inv(iC)




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
        self.dim = shape[0][0]
        self.N = 0;
        self.weights = np.ones((self.n,))
        self.weights /= self.n
        self.means = np.zeros((self.n,self.dim))
        for it in range(self.n):
            self.means[it] = dataset[it]
        self.covars = np.array([np.identity(self.dim) for x in range(self.n)])



    def __str__(self):
        out = ""
        np.set_printoptions(threshold=np.nan)
        out += 'w: ' + str(self.weights) + '\nm: ' + str(self.means) + '\nc: ' + str(self.covars)
        return out

    
