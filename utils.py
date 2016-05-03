import numpy as np
import os
from scipy import linalg
from config import *
EPS = np.finfo(float).eps



def genModels(n,dim,lmean, umean, lcovar, ucovar):
    models = []
    for i in range(n):
        mean = np.random.random((dim,)) * umean + lmean
        mat = np.random.random((dim,dim)) * ucovar + lcovar
        cov = (mat + mat.T)/2.0
        m = {
          'w': 1.0/n,
          'm': mean,
          'c': cov
         }
        models.append(m)
    return models



def mkdirs(config):
    for path in config.dirs:
        if not os.path.exists(path):
            os.mkdir(path)


