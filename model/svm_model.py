import config
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), config.LIBLINEAR_PATH))
print(sys.path)
from .model import Model
import liblinearutil as svm


class SVMModel(Model):
    def __init__(self, options="", identity=0):
        super().__init__(options=options)
        self.model = None

    def train(self, X, y):
        self.model = svm.train(y, X, self.options)

    def predict(self, X, y=None, options=""):
        label, _, value = svm.predict(y, X, self.model)
        return label, value

    def evaluate(self, true_y, pred):
        pass