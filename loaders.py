import  mnist 
import numpy as np
from config import *
from sklearn.datasets import fetch_20newsgroups

mnist_init_test = [7,0,1,2,3,4,8,11,18,61]
mnist_init_train = [0,1,2,3,4,5,7,13,15,17]

def mnist_loader(config):
    m = mnist.MNIST(path=config.dataset_params[PATH])
    m.load_testing()
    m.load_training()
    t = []
    lab = []
    ini=[]

    if config.dataset_params[SET] == TEST:
        t.extend(m.test_images)
        lab.extend(m.test_labels)
        if config.dataset_init == INIT_FIXED:
            for i in mnist_init_test:
                ini.append(m.test_images[i])


    if config.dataset_params[SET] == TRAIN:
        t.extend(m.train_images)
        lab.extend(m.train_labels)
        if config.dataset_init == INIT_FIXED:
            for i in mnist_init_train:
                ini.append(m.train_images[i])


    if config.dataset_params[SET] == TRAINTEST:
        t.extend(m.test_images)
        t.extend(m.train_images)
        lab.extend(m.test_labels)
        lab.extend(m.train_labels)
        if config.dataset_init == INIT_FIXED:
            for i in mnist_init_test:
                ini.append(m.test_images[i])


    if config.dataset_init == INIT_RANDOM or config.dataset_init == INIT_FIRST:
        ini=t[:config.alg_params[CLUSTERS]]


    return t, ini, lab

def covertype_loader(config):
    raw = []
    inivisited=[]
    ini = []
    labels=[]
    path=config.dataset_params[PATH]
    raw = np.load(path+"/data.npy")
    raw = raw.astype(np.float)
    labels = np.load(path+"/labels.npy")
    labels = labels.astype(np.int)

    if config.dataset_init == INIT_FIXED:
        for it,x in enumerate(labels[1]):
            if x not in inivisited:
                inivisited.append(x)
                ini.append(raw[it])
            if len(inivisited) == 7:
                break
    if config.dataset_init == INIT_RANDOM or config.dataset_init == INIT_FIRST:
        ini=raw[:config.alg_params[CLUSTERS]]

    return raw, ini, labels

def news_groups_loader(path):
    train = fetch_20newsgroups(subset='train')
    test = fetch_20newsgroups(subset='test')
