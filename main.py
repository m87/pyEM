import argparse
from utils import Model
from utils import DataStream
from utils import MultivariateGaussian
import utils
import online
import offline

algs = {
    "stepwise": online.Stepwise,
    "batch" : offline.Batch,
    "batch-entropy" : offline.BatchEntropy,
    "entropy" : online.Entropy,
    "incremental-one" : online.IncrementalOne,
    "incremental-k" : online.IncrementalK,
    "incremental-inf" : online.IncrementalInf,

}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algorithm', help='Algorithm')
    parser.add_argument('-p', '--param', help='Params', type=str)
    parser.add_argument('-n', '--size', help='Size', type=int)
    parser.add_argument('-c', '--clusters', help='Clusters', type=int)
    args = parser.parse_args()

    model = Model(2,2)
    model.set(0,0.5, ((20,50)), ((50,20),(20, 10)))
    model.set(1,0.5, ((12,10)), ((1,0),(0, 1)))

    ga = MultivariateGaussian(model)
    ar = ga.array(args.size * 2)
    stream = DataStream(src=ar, n=args.size, size=1)

    param = 0
    if args.param:
        if (args.algorithm.startswith('inc')):
            param = int(args.param)
        else:
            param = float(args.param)


    alg = algs[args.algorithm](args.clusters, param)
    models = alg.fit(stream)


    utils.display_result(models, ar)
    utils.display_err(alg.stats.error)

if __name__ == '__main__':
    main()
