import yaml

ALG = "alg"
RESULT_PATH = "result_path"
TYPE = "type"
ALPHA = "alpha"
SKIP = "skip"
CLUSTERS = "clusters"
PARAMS = "params"

PLOT = "plot"
PLOT_ERR = "error"
PLOT_2d = "2d_result"

DATASET="dataset"
INIT = "init"
N = "n"

WEIGHTS="weights"
MEANS="means"
COVARS = "covars"

FIXED_GEN = 'fixed-generator'
LIM_GEN = 'lim-generator'

MNIST = 'mnist'
COVERTYPE = 'covertype'
PATH = 'path'

INIT_RANDOM = 'random'
INIT_FIXED = 'fixed'
INIT_FIRST = 'first'
DIRS = 'mkdirs'
SET = 'set'
TRAIN = 'train'
TRAINTEST = 'train+test'
TEST = 'test'
NORM = 'norm'

class Config(object):
    def __init__(self, path):
        f = open(path, 'r')
        raw = yaml.load(f)

        self.alg_type = raw[ALG][TYPE]
        self.alg_params = raw[ALG][PARAMS]
        self.alg_result_path = raw[ALG][RESULT_PATH]
        self.plot_err = bool(raw[PLOT][PLOT_ERR])
        self.plot_2d = bool(raw[PLOT][PLOT_2d])
        self.dataset_type = raw[DATASET][TYPE]
        self.dataset_init = raw[DATASET][INIT]
        self.dataset_n = int(raw[DATASET][N])
        self.dataset_params = raw[DATASET][PARAMS]
        self.dirs =  raw[DIRS]
        self.predict = bool(raw['predict']['active'])
        self.predict_path = raw['predict']['model_path']

        f.close()
