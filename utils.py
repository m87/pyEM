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



def gen(models, size):
    clusters = len(models['weights'])
    out=[]
    ini = []
    labels =[]
    n= int(size/clusters)+1
    for m in range(clusters):
        w=np.random.multivariate_normal(models['means'][m],models['covars'][m],n)
        x = [[i,m] for i in w ]
        out.extend(x)
        ini.append(w[0])
    np.random.shuffle(out)
    out = tuple(zip(*out))
    return np.array(out[0]),ini, np.array(out[1])




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


