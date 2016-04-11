import utils as uts
import numpy as np

class Incremental(object):
    def __init__(self, n_clusters, select, k=20):
        self.n = n_clusters
        self.k = k
        self.select = select
        self.hist = []
        self.func = {
            'one': (self.__e_inf, self.__m_one),
            'inf': (self.__e_inf, self.__m_inf),
            'k': (self.__e_inf, self.__m_k),
        }

    def __e_one(self, X):
        pass
        
    def __e_inf(self, X):
        self.resps = self.weights * np.exp(uts.log_mvnpdf(np.array([X]), self.means, self.covars))
        self.resps = np.array(self.resps[0])
        self.hist.append(np.log(sum(self.resps)))
        np.clip(self.resps, 0.0000000000001, np.inf, out=self.resps)
        self.resps /= self.resps.sum()

    def __e_k(self, X):
        pass

    def __m_one(self,X):
        c = np.random.choice(np.arange(self.n), p=self.resps)
        self.N += 1
        self.accResps[c] += 1#self.resps[c]
        self.accMeans[c] += X# * self.resps[c]
        self.weights[c] = self.accResps[c] / self.N

        self.means[c] = self.accMeans[c] / self.accResps[c]

        diff = X - self.means[c]
        self.accCovars[c] +=  np.outer(diff, diff)

        self.covars[c] = self.accCovars[c] / self.accResps[c]



    def __m_inf(self,X):
        self.N += 1
        self.accResps += self.resps
        self.accMeans += X * self.resps[:,None]
        
        self.weights = self.accResps / self.N

        self.means = self.accMeans / self.accResps[:,None]

        for c in np.arange(self.n):
            diff = X - self.means[c]
            self.accCovars[c] +=  np.outer(self.resps[c] * diff, diff)

        self.covars = self.accCovars / self.accResps[:,None,None]



    def __m_k(self,X):
        K = np.ones((self.n))
        K /= 100000000
        C = np.random.choice(np.arange(self.n), p=self.resps, size=self.k)
        for c in C:
            K[c]+=1
        K/=self.k

        self.N += 1
        self.accResps += K
        self.accMeans += X * K[:,None]
        
        self.weights = self.accResps / self.N

        self.means = self.accMeans / self.accResps[:,None]

        for c in np.arange(self.n):
            diff = X - self.means[c]
            self.accCovars[c] +=  np.outer(K[c] * diff, diff)

        self.covars = self.accCovars / self.accResps[:,None,None]




    def __e(self,X):
        self.func[self.select][0](X)

    def __m(self,X):
        self.func[self.select][1](X)



    def fit(self, dataset):
        #print(np.exp(uts.log_mvnpdf(np.array([[1,1]]), np.array([[1,1]]), np.array([[[1,0],[0,1]]]))))
        #print(dataset.shape())
        self.__prepare(dataset)
        for it, X in dataset:
            self.__e(X)
            self.__m(X)



    def __prepare(self, dataset):
        shape = dataset.shape()
        dim = shape[0][0]
        self.N = 0;
        self.accResps = np.zeros((self.n,))
        self.accMeans = np.zeros((self.n,dim))
        self.accCovars = np.zeros((self.n,dim,dim))
        self.weights = np.ones((self.n,))
        self.weights /= self.n
        self.means = np.zeros((self.n,dim))
        for it in range(self.n):
            self.means[it] = dataset[it]
        self.covars = np.array([np.identity(dim) for x in range(self.n)])



    def __str__(self):
        out = ""
        np.set_printoptions(threshold=np.nan)
        out += 'w: ' + str(self.weights) + '\nm: ' + str(self.means) + '\nc: ' + str(self.covars)
        return out


    def save(self, path):
        np.save(path+"/weights", self.weights)
        np.save(path+"/means", self.means)
        np.save(path+"/covars", self.covars)
        np.save(path+"/hist", self.hist)
