import argparse
import os
from stepwise import Stepwise, StepwiseGauss
from entropy import Entropy
from incremental import Incremental
import utils
from dataset import Dataset, get_stream
from offline import Batch, BatchEntropy
from config import *
from benchmark import *

algs= {
    "stepwise": {
        "gauss" : StepwiseGauss,
    },
    "incremental":{
        "gauss" : None,#IncrementalGauss,
    },
    "entropy": {
        "gauss" : None,
    },
    "batch": {
        "gauss" : None,
    },
    "batch-entropy": {
        "gauss" : None,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Config file')
    args = parser.parse_args()

    config = Config(args.config)
    utils.mkdirs(config)
    stream = get_stream(config)

    alg = algs[config.alg_type][config.alg_subtype](config.alg_params)
    if not config.predict :
        alg.fit(stream)
        alg.save(config.alg_result_path)
        plot(config, alg, stream)
    else:
        alg.load(config.predict_path)
        result, RSL = alg.predict(stream)
        np.save('RSL', np.array(RSL))
        results(stream, result, config)


if __name__ == '__main__':
    main()
