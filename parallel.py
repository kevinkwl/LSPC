from joblib import Parallel, delayed


def parallel_train(train_func, Xs, Ys, n_jobs=8, testX=None):
    """
    train models in parallel (M3-network)
    
    :param train_func: the function to train a model, should accept three argument (X, y, identifier=(i,j))
    :param Xs: a list of list of X
    :param Ys: a list of list of Y
    :param n_jobs: train in n separate process (-1 means use all cpu cores)
    :return: a list of models (train_func return value)
    """
    minmodules = len(Xs[0])
    maxmodules = len(Xs)

    return Parallel(n_jobs=n_jobs)(delayed(train_func)(Xs[i][j],
                                                       Ys[i][j],
                                                       (i, j),
                                                       testX=testX)
                                   for i in range(maxmodules) for j in range(minmodules))
