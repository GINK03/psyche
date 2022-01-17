import numpy as np

def mape(ypred, ytrue):
    assert len(ypred) == len(ytrue), "do not match mape input size"

    N = len(ypred)
    c = 0
    for n in range(N):
        c += abs(ytrue[n]-ypred[n])/ytrue[n]
    mape = 100*c/N
    print(mape)
    return mape

def r2(ypred, ytrue):
    assert len(ypred) == len(ytrue), "do not match mape input size"

    N = len(ypred)
    y_mean = np.mean(ytrue)
    c1,c2 = 0,0
    for n in range(N):
        c1 += (ytrue[n]-ypred[n])**2
        c2 += (ytrue[n]-y_mean)**2
    ret = 1 - c1/c2
    print('in eval func', ret, c1, c2)
    return ret*-1
