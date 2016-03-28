import argparse
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2
from sklearn import mixture as mix
import numpy as np

from utils import Model
from utils import DataStream
from utils import MultivariateGaussian
import online
import offline

algs = {
    "stepwise": online.Stepwise,
    "batch" : offline.Batch
}



def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def display_result(model, instances):
    plt.plot()
    plt.title('model')

    i = list(zip(*instances))
    plt.scatter(x=i[0], y=i[1])
    ax = None
    if ax is None:
        ax = plt.gca()

    for mo in range(model.n):
        vals, vecs = eigsorted(model.covars[mo])
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        width, height = 2 * np.sqrt(chi2.ppf(0.5, 2)) * np.sqrt(vals)
        ellip = Ellipse(xy=model.means[mo,0], width=width, height=height, angle=theta, alpha=0.5, color='red')

        ax.add_artist(ellip)

    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algorithm', help='Algorithm')
    parser.add_argument('-p', '--param', help='Params', type=str)
    parser.add_argument('-n', '--size', help='Size', type=int)
    parser.add_argument('-c', '--clusters', help='Clusters', type=int)
    args = parser.parse_args()

    # ga = MultivariateGaussian((Model(2, (2, 5), ((5, 2), (2, 1))), Model(2, (12, 10), ((1, 0), (0, 1))),
    #                           Model(2, (0, 1), ((1, 0), (0, 1)))))


    model = Model(2,2)
    model.set(0,0.5, ((20,50)), ((50,20),(20, 10)))
    model.set(1,0.5, ((12,10)), ((1,0),(0, 1)))

    ga = MultivariateGaussian(model)
    ar = ga.array(args.size * 2)
    stream = DataStream(src=ar, n=args.size, size=1)

    param = 0
    if args.param:
        if (args.algorithm.startswith('inc')):
            param = int(args.param)
        else:
            param = float(args.param)


    alg = algs[args.algorithm](args.clusters, param)
    models = alg.fit(stream)


    display_result(models, ar)
    plt.plot(list(zip(*alg.history().all()))[0])
    plt.show()


if __name__ == '__main__':
    main()
