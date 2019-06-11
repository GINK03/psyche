
import pandas as pd

def lgb(models=[], train_columns=[]):
    c_i = {}
    for idx, model in enumerate(models):
        for c, i in zip(train_columns, model.feature_importance(importance_type='gain')):
            if c_i.get(c) is None:
                c_i[c] = 0
            c_i[c]+=i
    dfImps = pd.DataFrame({'cs':list(c_i.keys()), 'is':list(c_i.values())})
    dfImps = dfImps.sort_values(by=['is'], ascending=False).reset_index()
    dfImps = dfImps.drop(['index'], axis=1)
    return dfImps


