import time
import numpy as np
import config


class TrainWrapper(object):
    """
    A wrapper for training models
    """
    def __init__(self, ModelClass, options=""):
        self.ModelClass = ModelClass
        self.options = options

    def __call__(self, X, y, identity=0, testX=None):
        start = time.time()
        model = self.ModelClass(self.options)
        try:
            model.train(X, y)
        except ValueError:
            print("Error !!!!!!")
            import sys
            sys.exit(-1)

        print("training model {} took {:.2f} seconds\n".format(identity, time.time() - start))

        if testX is not None:
            start = time.time()
            label, val = model.predict(testX)
            print("model {} took {:.2f} seconds to predict\n".format(identity, time.time() - start))
            return label, val

        if config.SAVE_MODEL:
            #save model
            pass
        return identity


def minmax(pred_val, pred_label, minmax_shape):
    # timer
    start = time.time()

    Nmax = minmax_shape[0]
    Nmin = minmax_shape[1]

    pred_val = np.reshape(pred_val, (Nmax, Nmin, -1))
    pred_label = np.reshape(pred_label, (Nmax, Nmin, -1))

    max_in_val = []
    max_in_label = []

    # min modules
    for i in range(Nmax):
        min_in_vals = []
        min_in_labels = []
        for j in range(Nmin):
            labels = pred_label[i][j]
            vals = pred_val[i][j]
            min_in_vals.append(vals)
            min_in_labels.append(labels)
        min_val = np.min(min_in_vals, axis=0)
        max_in_val.append(min_val)

        min_label = np.min(min_in_labels, axis=0)
        max_in_label.append(min_label)

    # max modules
    output_val = np.max(max_in_val, axis=0)
    output_label = np.max(max_in_label, axis=0)

    end = time.time()
    print("minmax took: {:.2f}s".format(end - start))

    return output_label, output_val
