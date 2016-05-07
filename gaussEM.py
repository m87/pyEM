import numpy as np 

class GaussEM():
    def load(self, path):
        self.weights = np.load(path+"/weights.npy")
        self.means = np.load(path+"/means.npy")
        self.covars = np.load(path+"/covars.npy")
        self.diagCovars = np.zeros((np.shape(self.covars[0])[0],))
        for c in self.covars:
            self.diagCovars[c] = np.diag(self.covars[c])



    def save(self, path):
        np.save(path+"/weights", self.weights)
        np.save(path+"/means", self.means)
        np.save(path+"/covars", self.covars)
        np.save(path+"/hist", self.hist)

    def __str__(self):
        out = ""
        np.set_printoptions(threshold=np.nan)
        out += 'w: ' + str(self.weights) + '\nm: ' + str(self.means) + '\nc: ' + str(self.covars)
        return out


