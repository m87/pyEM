from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2
import numpy as np


def results(stream, result):
    pass

def display_err(err):
    #print(err)
    #for it, i in enumerate(err):
    #    err[it]= i/(it+1)
    plt.plot(err[1:])
    #plt.yscale('log')
    plt.show()
    #plt.savefig('./model/plot.pdf')

def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def display_result(mu,cov, instances):
    if(np.shape(mu[0]) != (2,)):
        return
    plt.plot()
    plt.title('model')

    i = list(zip(*instances))
    plt.scatter(x=i[0], y=i[1])
    ax = None
    if ax is None:
        ax = plt.gca()

    for mo in range(len(mu)):
        vals, vecs = eigsorted(cov[mo])
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        width, height = 2 * np.sqrt(chi2.ppf(0.5, 2)) * np.sqrt(vals)
        ellip = Ellipse(xy=mu[mo], width=width, height=height, angle=theta, alpha=0.5, color='red')

        ax.add_artist(ellip)

    plt.show()



def plot(config, alg, stream):
    if config.plot_2d:
        display_result(alg.means, alg.covars, stream[:])
    if config.plot_err:
        display_err(alg.hist)

