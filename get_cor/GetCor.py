import numpy as np
import pandas as pd

def calc_with_specific_key(key='y', df=None):
    ys = df[key]

    objs = []
    for c in df.columns:
        xs = df[c]
        cor = np.corrcoef(xs, ys)[0,1]
        obj = {'colname':c, 'cor_abs':np.abs(cor), 'cor':cor}
        objs.append(obj)

    dfRet = pd.DataFrame(objs).sort_values('cor_abs', ascending=False)
    return dfRet
    
