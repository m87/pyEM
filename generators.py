import numpy as np
import os
from scipy import linalg
from config import *



def fixed_generator(models, size, init):
    clusters = len(models[WEIGHTS])
    out=[]
    ini = []
    labels =[]
    n= int(size/clusters)+1
    for m in range(clusters):
        w=np.random.multivariate_normal(models[MEANS][m],models[COVARS][m],n)
        x = [[i,m] for i in w ]
        out.extend(x)
        ini.append(w[0])
    out = tuple(zip(*out))

    if init == INIT_RANDOM:
        a = np.random.choice(range(len(out[0])), clusters)
        ini =[]
        for i in a:
            ini.append(out[0][i])
    return np.array(out[0]),ini, np.array(out[1])




def lim_generator():
    pass
