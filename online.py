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
        for it, X in dataset:
            print(it)
            self.e(X)
            self.m(X)


    def prepare(self, dataset):
        shape = dataset.shape()
        dim = shape[0][0]
        self.dim=dim
        self.N = 0;

    def predict(self, dataset):
        self.prepare(dataset)
        res = np.zeros((self.n,))
        resL = [[] for i in range(self.n)]
        for it, X in dataset:
            print(it)
            self.e(X)
            res[np.argmax(self.resps)] +=1
            resL[np.argmax(self.resps)].append(dataset.label())

        return res, resL





