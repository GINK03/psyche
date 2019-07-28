
import pandas as pd
import numpy as np

def clip(input_dfs=[], pecentage=1, ignores=None):
    assert isinstance(ignores, list), "ignores must be a list of column-name."
    dfTrain, dfTest = input_dfs
    size = len(dfTrain)

    dfConc = pd.concat([dfTrain, dfTest], axis=0)

    for c in dfConc.columns:
        if c in ignores:
            continue
        upper, lower = np.percentile(dfConc[c], [percentage, 100-percentage])
        dfConc[c] = np.clip(dfConc[c], upper, lower)


    dfTrain = dfConc[:size]
    dfTest = dfConc[size:]
    return dfTrain, dfTest
