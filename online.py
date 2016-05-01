import utils as uts
import numpy as np
import os
from config import *

mvnpdf = {'full': uts.log_mvnpdf, 'diag': uts.log_mvnpdf_diag}
EPS = np.finfo(float).eps

class OnlineEM():
    def __init__(self, param):
        self.n = param[CLUSTERS]
        self.hist = []
        self.histAcc = 0.0

    def e(self, X):
        pass
    def m(self, X):
        pass

    def fit(self, dataset):
        #print(np.exp(uts.log_mvnpdf(np.array([[1,1]]), np.array([[1,1]]), np.array([[[1,0],[0,1]]]))))
        #print(dataset.shape())
        self.prepare(dataset)
        for it, X in dataset:
            print(it)
            self.e(X)
            self.m(X)



    def prepare(self, dataset):
        shape = dataset.shape()
        dim = shape[0][0]
        self.dim=dim
        self.N = 0;
        self.accResps = np.zeros((self.n,))
        self.accMeans = np.zeros((self.n,dim))
        self.accCovars = np.zeros((self.n,dim,dim))
        self.weights = np.ones((self.n,))
        self.weights /= self.n
        self.means = np.zeros((self.n,dim))
        for it,x in enumerate(dataset.getInit()):
            self.means[it] = x
        self.covars = np.array([np.identity(dim) for x in range(self.n)])
        self.diagCovars = np.ones((self.n,dim))
        self.COV = {'full' : self.covars, 'diag' : self.diagCovars}
        self.I ={'full': 1.0, 'diag': np.identity(self.dim)}


    def load(self, path):
        self.weights = np.load(path+"/weights.npy")
        self.means = np.load(path+"/means.npy")
        self.covars = np.load(path+"/covars.npy")

    def save(self, path):
        np.save(path+"/weights", self.weights)
        np.save(path+"/means", self.means)
        np.save(path+"/covars", self.covars)
        np.save(path+"/hist", self.hist)

    def __str__(self):
        out = ""
        np.set_printoptions(threshold=np.nan)
        out += 'w: ' + str(self.weights) + '\nm: ' + str(self.means) + '\nc: ' + str(self.covars)
        return out


