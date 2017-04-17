from .model import Model
import liblinearutil as svm


class SVMModel(Model):
    def __init__(self, options="", identity=0):
        super().__init__(options=options)
        self.model = None

    def train(self, X, y):
        self.model = svm.train(y, X, self.options)

    def predict(self, X, y=None, options=""):
        svm.predict()
        pass

    def evaluate(self, true_y, pred):
        pass