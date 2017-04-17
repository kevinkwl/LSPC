import time
import numpy as np


class TrainWrapper(object):
    """
    A wrapper for training models
    """
    def __init__(self, ModelClass, options=""):
        self.ModelClass = ModelClass
        self.options = options

    def __call__(self, X, y, identity=0):
        start = time.time()
        model = self.ModelClass(self.options, identity)
        model.train(X, y)
        print("model {} took {:.2f} seconds\n".format(identity, time.time() - start))
        return model


class PredictWrapper(object):
    """
    A wrapper for predicting, min-max
    """
    def __init__(self, models, options=None):
        """
        
        :param models: an i x j list of models, 
        :param options: predicting options
        """
        assert isinstance(models, list) and isinstance(models[0], list), "models should be 2-dimensional list"
        self.models = models
        self.options = options

    def __call__(self, X, y=None):
        Nmax = len(self.models)
        Nmin = len(self.models[0])

        max_in_val = []
        max_in_label = []

        # min modules
        for i in range(Nmax):
            min_in_vals = []
            min_in_labels = []
            for j in range(Nmin):
                labels, vals = self.models[i][j].predict(X, y, self.options)
                min_in_vals.append(labels)
                min_in_labels.append(vals)
            min_val = np.min(min_in_vals, axis=0)
            max_in_val.append(min_val)

            min_label = np.min(min_in_labels, axis=0)
            max_in_label.append(min_label)

        # max modules
        output_val = np.max(max_in_val, axis=0)
        output_label = np.max(max_in_label, axis=0)

        return output_label, output_val
