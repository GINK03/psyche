import pandas as pd
import statistics
import random

def encode_with_columns_mutable_2(input_dfs=[], target=None, columns=[]):
    dfTrain, dfTest = input_dfs
    dfConcat = pd.concat([dfTrain, dfTest], axis=0) 
    ys = dfConcat[target]
    for c in columns:
        test_selected = set(dfTest[c].tolist())
        selected = dfConcat
        s_cs = {}
        for s, y in zip(selected, ys):
            if s_cs.get(s) is None:
                s_cs[s] = []
            s_cs[s].append(y)

        for s in list(s_cs.keys()):
            if s in test_selected:
                s_cs[s] = statistics.mean(s_cs[s])
            else:
                s_cs[s] = 0

        for df in [dfTrain, dfTest]:
            df[f'TargetEncoding_{c}'] = df[c].apply(lambda s: s_cs[s] if s_cs.get(s) else 0 )

def noise(x):
    ra = random.random()*0.2
    return (0.8+ra)*x

def encode_with_columns_mutable(input_dfs=[], ys=None, columns=[]):
    dfTrain, dfTest = input_dfs

    for c in columns:
        test_selected = set(dfTest[c].tolist())
        selected = dfTrain[c]

        s_cs = {}
        for s, y in zip(selected, ys):
            if s_cs.get(s) is None:
                s_cs[s] = []
            s_cs[s].append(y)

        for s in list(s_cs.keys()):
            if s in test_selected:
                s_cs[s] = statistics.mean(s_cs[s])
            else:
                s_cs[s] = 0

        for df in [dfTrain, dfTest]:
            df[f'TargetEncoding_{c}'] = df[c].apply(lambda s: noise(s_cs[s]) if s_cs.get(s) else 0)
