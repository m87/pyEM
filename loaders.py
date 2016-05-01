import  mnist 
import numpy as np
from config import *
from sklearn.datasets import fetch_20newsgroups

mnist_init_test = [7,0,1,2,3,4,8,11,18,61]
mnist_init_train = [0,0,0,0,0]

def mnist_loader(config):
    m = mnist.MNIST(path=config.dataset_params[PATH])
    m.load_testing()
    m.load_training()
    t = []
    lab = []

    if config.dataset_params[SET] == TEST:
        t.extend(m.test_images)
        lab.extend(m.test_labels)
        if config.dataset_init == INIT_FIXED:
            ini=[m.test_images[7],m.test_images[0],m.test_images[1],m.test_images[2],m.test_images[3],m.test_images[4],m.test_images[8],m.test_images[11],m.test_images[18],m.test_images[61]]


    if config.dataset_params[SET] == TRAIN:
        t.extend(m.train_images)
        lab.extend(m.train_labels)
        if config.dataset_init == INIT_FIXED:
            ini=[m.train_images[0],m.train_images[1],m.train_images[2],m.train_images[3],m.train_images[4],m.train_images[5],m.train_images[7],m.train_images[13],m.train_images[15],m.train_images[17]]


    if config.dataset_params[SET] == TRAINTEST:
        t.extend(m.test_images)
        t.extend(m.train_images)
        lab.extend(m.test_labels)
        lab.extend(m.train_labels)
        if config.dataset_init == INIT_FIXED:
            ini=[m.test_images[7],m.test_images[0],m.test_images[1],m.test_images[2],m.test_images[3],m.test_images[4],m.test_images[8],m.test_images[11],m.test_images[18],m.test_images[61]]




    np.random.shuffle(t)
    if config.dataset_init == INIT_RANDOM or config.dataset_init == INIT_FIRST:
        ini=t[:config.alg_params[CLUSTERS]]


    return t, ini, labels

def covertype_loader(path):
    pass


def news_groups_loader(path):
    train = fetch_20newsgroups(subset='train')
    test = fetch_20newsgroups(subset='test')
