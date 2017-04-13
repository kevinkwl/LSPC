from model.svm_model import SVMModel
from train import TrainWrapper
from parallel import parallel_train
from liblinearutil import svm_read_problem

y, X = svm_read_problem("files/test")
train_func = TrainWrapper(SVMModel)

parallel_train(train_func, Xs=[X], Ys=[y], n_jobs=1)




