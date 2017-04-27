from sklearn.neural_network import MLPClassifier
from .model import Model
from scipy.sparse import vstack

class Config:
    hidden = 100
    verbose = True
    max_iter = 10

class MLPModel(Model):
    def __init__(self, options=""):
        super().__init__(options)
        self.model = MLPClassifier(hidden_layer_sizes=Config.hidden, verbose=Config.verbose, max_iter=Config.max_iter)
    def train(self, X, y):
        # need to transform the input data after dividing (violates the csr_matrix)
        X = vstack(X, "csr")
        self.model.fit(X, y)

    def predict(self, X, y=[], options=""):
        label = self.model.predict(X)
        val = self.model.predict_proba(X)
        select = 0 if self.model.classes_[0] == 1 else 1
        val = [v[select] for v in val]
        return label, val