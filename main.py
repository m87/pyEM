import argparse
import os
from stepwise import Stepwise
from entropy import Entropy
from incremental import Incremental
import utils
from dataset import Dataset, get_stream
from offline import Batch, BatchEntropy
from config import *
from benchmark import *

algs = {
    "stepwise": Stepwise,
    "incremental": Incremental,
    "entropy": Entropy,
    "batch-entropy": BatchEntropy,
    "batch": Batch,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Config file')
    args = parser.parse_args()

    config = Config(args.config)
    utils.mkdirs(config)
    stream = get_stream(config)


    alg = algs[config.alg_type](config.alg_params)
    alg.fit(stream)
    alg.save(config.alg_result_path)

    plot(config, alg, stream)

if __name__ == '__main__':
    main()
