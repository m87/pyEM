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

    t = list(zip(t,lab))
    np.random.shuffle(t)
    t = tuple(zip(*t))

    if config.dataset_init == INIT_RANDOM or config.dataset_init == INIT_FIRST:
        ini=t[0][:config.alg_params[CLUSTERS]]


    return t[0], ini, t[1]

def covertype_loader(path):
    pass


def news_groups_loader(path):
    train = fetch_20newsgroups(subset='train')
    test = fetch_20newsgroups(subset='test')
