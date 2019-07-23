import json
import glob
import sys
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import KFold, KFold
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import EvalFunc
import xgboost as xgb
print('load xgb train')
def shot_train(xs, ys, XT, param, folds, eval_func, early_stopping_rounds):
    if isinstance(xs, (pd.DataFrame)):
        print('input xs, ys, XT may be pd.DataFrame, so change to np.array')
        xs = xs.values
        ys = ys.values
        XT = XT.values
    assert xs.shape[1] == XT.shape[1], "column size is mimatach!!"
    eval_losses = []
    oof_predictions = np.zeros((len(xs),))
    test_predictions = np.zeros((len(XT),))
    models = []
    for idx, (trn_idx, val_idx) in enumerate(folds.split(xs, ys)):
        print(f'now fold={idx:02d} split size is', folds.get_n_splits())
        #trn_data = lgb.Dataset(xs[trn_idx], label=ys[trn_idx], categorical_feature=category)
        #val_data = lgb.Dataset(xs[val_idx], label=ys[val_idx], categorical_feature=category)
        trn_data = xgb.DMatrix(xs[trn_idx], ys[trn_idx])
        val_data = xgb.DMatrix(xs[val_idx], ys[val_idx])
        clf = xgb.train(params=param, dtrain=trn_data, num_boost_round=200, evals=[(trn_data, "Train"), (val_data, "Val")],
                 verbose_eval=100, early_stopping_rounds=100)

        oof_predictions[val_idx] = clf.predict(xgb.DMatrix(xs[val_idx]))
        test_predictions += clf.predict(xgb.DMatrix(XT)).flatten() / folds.get_n_splits()
        eval_loss = eval_func(ys[val_idx], clf.predict(xgb.DMatrix(xs[val_idx])))
        print(f'end fold={idx:02d} eval_loss={eval_loss}')
        eval_losses.append(eval_loss)
        models.append(clf)

    return {'eval_loss': np.mean(eval_losses),
            'oof_predictions': oof_predictions,
            'test_predictions': test_predictions,
            'models': models}
