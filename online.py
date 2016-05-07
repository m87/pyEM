import utils as uts
import numpy as np
import os
from config import *

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
        self.prepare(dataset)
        for it, X in enumerate(dataset):
            print(it)
            self.e(X)
            self.m(X)


    def prepare(self, dataset):
        self.dim, _ = dataset.shape()
        self.dim = self.dim[0]
        self.N =1

    def predict(self, dataset):
        self.prepare(dataset)
        res = np.zeros((self.n,))
        resL = [[] for i in range(self.n)]
        for it, X in enumerate(dataset):
            print(it)
            self.e(X)
            res[np.argmax(self.resps)] +=1
            resL[np.argmax(self.resps)].append(dataset.L[it])

        return res, resL

    #TODO macierz pomyłek



