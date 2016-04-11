import argparse
from stepwise import Stepwise
from entropy import Entropy
from incremental import Incremental
import utils
from dataset import Dataset
from  mnist import MNIST
from offline import Batch

algs = {
    "stepwise": Stepwise,
    "incremental": Incremental,
    "entropy": Entropy,
    "batch": Batch,

}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algorithm', help='Algorithm')
    parser.add_argument('-p', '--param', help='Params', type=str)
    parser.add_argument('-n', '--size', help='Size', type=int)
    parser.add_argument('-c', '--clusters', help='Clusters', type=int)
    parser.add_argument('-m', '--mnist', help='Clusters', type=bool)
    args = parser.parse_args()

    model = []
    m1 = {
          'w': 0.5,
          'm': (2,5),
          'c': ((5,2),(2,1))
         }
    m2 = {
          'w': 0.5,
          'm': (12,10),
          'c': ((1,0),(0,1))
         }

    model.append(m1)
    model.append(m2)


    if args.mnist:
        m = MNIST("./data/mnist")
        m.load_testing()
        m.load_training()
        t = []
        t.extend(m.test_images)
        t.extend(m.train_images)
        stream = Dataset(src=t, n=args.size, size=1)
    else:
        ar = utils.gen(model, args.size)
        stream = Dataset(src=ar, n=args.size, size=1)

    #print(">>training")
    alg = algs[args.algorithm](args.clusters, args.param)
    alg.fit(stream)
    #alg.save("./model")
    print(alg)

    utils.display_err(alg.hist)



if __name__ == '__main__':
    main()
