from parallel import parallel_train
from wrapper import TrainWrapper, minmax
from utils import *
import config
from divide import divide
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), config.LIBLINEAR_PATH))
import numpy as np


def run(Model, module_size=0, option="random", scipy=False):
    """
    run liblinear training    
    :param module_size: the data module for size for each model, 0 for no decomposition
    :param option: class, nothing, random
    :return: 
    """
    print("Reading data...")
    y, X, tag = readData(config.TRAIN_FILE, return_scipy=scipy)
    testY, testX, _ = readData(config.TEST_FILE, return_scipy=scipy)
    train = TrainWrapper(Model, config.LIBLINEAR_TRAINING_OPTIONS)
    print("Reading data completed.")

    if module_size > 0:
        sort_tag = 0 if option == 'class' else (1 if option == 'nothing' else 2)
        posXs, negXs = divide(tag, X, module_size, sort_tag=sort_tag)
        print("Dividing completed.")
        Xs, Ys = getData(posXs, negXs)
        minmax_shape = (len(posXs), len(negXs))
        print("minmax shape: {}".format(minmax_shape))
        plabels, pvals = list(zip(*parallel_train(train, Xs, Ys, testX=testX)))
        plabel, pval = minmax(pvals, plabels, minmax_shape)
    else:
        plabel, pval = train(X, y, testX=testX)

    # analysis
    assert len(plabel) == len(testY)
    total = len(testY)
    hit = 0
    for idx, label in enumerate(plabel):
        if testY[idx] == label:
            hit += 1

    print("Accuracy = {:.2f} ({}/{})".format(hit*100/total, hit, total))

    saveResult("{}-{}-{}".format(Model.__name__, option, module_size), plabel, pval)


if __name__ == '__main__':
    import sys
    method = sys.argv[1]
    scipy = False
    if method == 'svm':
        from model.svm_model import SVMModel
        model = SVMModel
    elif method == 'mlp':
        from model.MLP_model import MLPModel
        model = MLPModel
        scipy = True
    msize = int(sys.argv[2])
    option = 'random'
    if len(sys.argv) > 3:
        option = sys.argv[3]
    run(model, msize, option, scipy)
