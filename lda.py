from mnist import MNIST
import numpy as np
from thirdparty import log_mvnpdf, log_mvnpdf_diag

data = MNIST('./data/mnist')
data.load_training()
data.load_testing()
train = np.array(data.train_images)/255.0
test = np.array(np.array(data.test_images)/255.0)
dataset = {i: [] for i in range(10) }

for it, x in enumerate(data.train_labels):
    dataset[x].append(train[it])

mu = []
cov = []
for k in dataset:
    mu.append(np.average(np.array(dataset[k])))
    cov.append(np.cov(np.array(dataset[k]).T))


es = log_mvnpdf(train, np.array(mu), np.array(cov)) 

results = {i: [] for i in range(10) }
for it,e in enumerate(es):
    results[np.argmax(e)].append(data.train_labels[it])

print(results)
