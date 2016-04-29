import numpy as np
import os
from scipy import linalg
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2
from config import *
EPS = np.finfo(float).eps



class VecError(Exception): pass

def vec_len(src):
    return len(src[0])

def vec(src):
    if len(np.shape(src)) == 2 and np.shape(src)[0] == 1:
        return np.array(src)
    if len(np.shape(src)) == 1:
        return np.array([src])
    raise VecError()

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



def gen(models, size):
    clusters = len(models['weights'])
    out=[]
    ini = []
    n= int(size/clusters)+1
    for m in range(clusters):
        w=np.random.multivariate_normal(models['means'][m],models['covars'][m],n)
        out.extend(w)
        ini.append(w[0])
        np.random.shuffle(out)
    return np.array(out),ini




# from scikit-learn https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/mixture/gmm.py
def log_mvnpdf(X, means, covars, min_covar=1.e-7):
    """Log probability for full covariance matrices."""
    n_samples, n_dim = X.shape
    nmix = len(means)
    log_prob = np.empty((n_samples, nmix))
    for c, (mu, cv) in enumerate(zip(means, covars)):
        try:
            cv_chol = linalg.cholesky(cv, lower=True)
        except linalg.LinAlgError:
            # The model is most probably stuck in a component with too
            # few observations, we need to reinitialize this components
            try:
                cv_chol = linalg.cholesky(cv + min_covar * np.eye(n_dim),
                                          lower=True)
            except linalg.LinAlgError:
                raise ValueError("'covars' must be symmetric, "
                                 "positive-definite")

        cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
        cv_sol = linalg.solve_triangular(cv_chol, (X - mu).T, 
lower=True).T
        log_prob[:, c] = - .5 * (np.sum(cv_sol ** 2, axis=1) +
                                 n_dim * np.log(2 * np.pi) + cv_log_det)

    return log_prob



def log_mvnpdf_diag(X, means, covars):
    """Compute Gaussian log-density at X for a diagonal model"""
    n_samples, n_dim = X.shape
    lpr = -0.5 * (n_dim * np.log(2 * np.pi) + np.sum(np.log(covars), 1)
                  + np.sum((means ** 2) / covars, 1)
                  - 2 * np.dot(X, (means / covars).T)
                  + np.dot(X ** 2, (1.0 / covars).T))
    return lpr



def mkdirs(config):
    for path in config.dirs:
        if not os.path.exists(path):
            os.mkdir(path)



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


