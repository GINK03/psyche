
def mape(ypred, ytrue):
    assert len(ypred) == len(ytrue), "do not match mape input size"

    N = len(ypred)
    c = 0
    for n in range(N):
        c += abs(ytrue[n]-ypred[n])/ytrue[n]
    mape = 100*c/N
    print(mape)
    return mape
