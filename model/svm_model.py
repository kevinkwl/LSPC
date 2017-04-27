import config
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), config.LIBLINEAR_PATH))
from .model import Model
import liblinearutil as svm


class SVMModel(Model):
    def __init__(self, options=""):
        Model.__init__(self, options=options)
        self.model = None

    def train(self, X, y):
        self.model = svm.train(y, X, self.options)

    def predict(self, X, y=[], options=""):
        label, _, value = svm.predict(y, X, self.model)
        return label, [v[0] for v in value]

    def evaluate(self, true_y, pred):
        pass