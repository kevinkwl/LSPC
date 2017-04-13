import time


class TrainWrapper(object):
    """
    A wrapper for training models
    """
    def __init__(self, ModelClass, options=""):
        self.ModelClass = ModelClass
        self.options = options

    def __call__(self, X, y, identity=0):
        self.model = self.ModelClass(self.options)
        start = time.time()
        model = self.ModelClass(self.options)
        model.train(X, y)
        print("model {} took {:.2f} seconds\n".format(identity, time.time() - start))
        return identity, model