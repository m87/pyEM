import argparse
import os
from stepwise import Stepwise, StepwiseGauss
from entropy import EntropyGauss
from incremental import IncrementalGauss
import utils
from dataset import Dataset, get_dataset
from offline import BatchGauss, BatchEntropy
from config import *
from benchmark import *

algs= {
    "stepwise": {
        "gauss" : StepwiseGauss,
    },
    "incremental":{
        "gauss" : IncrementalGauss,
    },
    "entropy": {
        "gauss" : EntropyGauss,
    },
    "batch": {
        "gauss" : BatchGauss,
    },
    "batch-entropy": {
        "gauss" : None,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Config file')
    args = parser.parse_args()

    np.set_printoptions(threshold=np.nan)
    config = Config(args.config)
    utils.mkdirs(config)
    dataset = get_dataset(config)
    #dataset.randomize()
    print(dataset[0], dataset.L[0])


    alg = algs[config.alg_type][config.alg_subtype](config.alg_params)
    if not config.predict :
        alg.fit(dataset)
        alg.save(config.alg_result_path)
        plot(config, alg, dataset)
    else:
        alg.load(config.predict_path)
        result, RSL = alg.predict(dataset)
        np.save('RSL', np.array(RSL))
        results(dataset, result, config)


if __name__ == '__main__':
    main()
