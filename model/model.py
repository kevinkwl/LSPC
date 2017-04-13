class Model(object):
    def __init__(self, options):
        self.options = options

    def train(self, X, y):
        pass

    def predict(self, X, y=None):
        pass

    def evaluate(self, true_y, pred):
        pass
