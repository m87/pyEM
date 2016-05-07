import numpy as np
import sys
raw =[]
labels=[]
with open(sys.argv[1], 'r') as f:
    for line in f.readlines():
        line = line.split(',')
        raw.append(line[:-1])
        labels.append(line[-1])

raw = np.array(raw)
labels = np.array(labels)

np.save("./data", raw)
np.save("./labels", labels)
