import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2



class VecError(Exception): pass

def vec_len(src):
    return len(src[0])

def vec(src):
    if len(np.shape(src)) == 2 and np.shape(src)[0] == 1:
        return np.array(src)
    if len(np.shape(src)) == 1:
        return np.array([src])
    raise VecError()


def gen(models, size):
    out=[]
    n= int(size/len(models))
    for m in models:
        out.extend(np.random.multivariate_normal(m['m'],m['c'],n))
    return np.array(out)




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



def display_err(err):
    print(err)
    for it, i in enumerate(err):
        err[it]= -i/(it+1)
    plt.plot(err)
    #plt.yscale('log')
    plt.savefig('./model/plot.pdf')


