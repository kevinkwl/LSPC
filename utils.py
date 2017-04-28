"""
utilities: 

"""

import config
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), config.LIBLINEAR_PATH))

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


def readData(file, return_scipy=False):
    from liblinearutil import svm_read_problem
    y, X = svm_read_problem(file + ".svm", return_scipy=return_scipy)
    tag  = open(file + ".tag").readlines()
    return y, X, tag

from config import SAVED
def saveResult(filename, pred_label, pred_value):
    with open(SAVED + filename, "w") as saved:
        for (label, value) in zip(pred_label, pred_value):
            saved.write("{} {}\n".format(label, value))

def readResult(filename):
    def splitit(line):
        s = line.split()
        return int(float(s[0])), float(s[1])
    with open(SAVED + filename, "r") as saved:
        zipped = [splitit(line) for line in saved]
        return list(zip(*zipped))

def readOracle():
    with open(config.TEST_FILE + ".svm") as of:
        true_label = [int(float(line.split()[0])) for line in of.readlines()]
    return true_label


from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve, accuracy_score

def score(true_label, pred_label):
    acc = accuracy_score(true_label, pred_label)
    precision = precision_score(true_label, pred_label)
    recall = recall_score(true_label, pred_label)
    f1 = f1_score(true_label, pred_label)
    return acc, precision, recall, f1