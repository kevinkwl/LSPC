from parallel import parallel_train
from wrapper import TrainWrapper, minmax
from utils import getData, readData
import config
from divide import divide

import numpy as np


def run_liblinear(module_size=-1, option="random"):
    """
    run liblinear training    
    :param module_size: the data module for size for each model, -1 for no decomposition
    :param option: class, nothing, random
    :return: 
    """
    from model.svm_model import SVMModel
    y, X, tag = readData(config.TRAIN_FILE)
    testY, testX, _ = readData(config.TEST_FILE)
    train = TrainWrapper(SVMModel, config.LIBLINEAR_TRAINING_OPTIONS)

    if module_size > 0:
        sort_tag = 0 if option == 'class' else (1 if option == 'nothing' else 2)
        posXs, negXs = divide(tag, X, module_size, sort_tag=sort_tag)
        print("Dividing completed.")
        Xs, Ys = getData(posXs, negXs)
        minmax_shape = (len(posXs), len(negXs))

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

if __name__ == '__main__':
    import sys
    msize = int(sys.argv[1])
    option = 'random'
    if len(sys.argv) > 2:
        option = sys.argv[2]
    run_liblinear(msize, option)