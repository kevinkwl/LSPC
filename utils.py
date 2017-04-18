"""
utilities: 

"""

import config
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), config.LIBLINEAR_PATH))


def plotroc(true_label, pval):
    """
    
    :param true_label: 
    :param pval: 
    :return: 
    """
    pass


def getData(posXs, negXs, posTag=1, negTag=-1):
    Nmin = len(negXs)
    Nmax = len(posXs)

    Xdata = []
    Ydata = []
    for i in range(Nmax):
        posX = posXs[i]
        Xdata.append([])
        Ydata.append([])
        for j in range(Nmin):
            negX = negXs[j]
            Xdata[i].append(posX + negX)
            Ydata[i].append([posTag] * len(posX) + [negTag] * len(negX))
    return Xdata, Ydata


def readData(file):
    from liblinearutil import svm_read_problem
    y, X = svm_read_problem(file + ".svm")
    tag  = open(file + ".tag").readlines()
    return y, X, tag