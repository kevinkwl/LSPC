from parallel import parallel_train

def stub_train_func(X, y, id):
    for i in range(1000000):
        pass

    print(id, "ended.")
    return id


Xs = [[x for x in range(100)] for i in range(20)]
Ys = [[y for y in range(100)] for i in range(20)]

print(parallel_train(stub_train_func, Xs, Ys))
