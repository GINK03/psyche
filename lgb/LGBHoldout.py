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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import EvalFunc


def shot_train(xs, ys, XT, category, param, folds, eval_func, verbose, early_stopping_rounds, n_estimators):
    if isinstance(xs, (pd.DataFrame)):
        print('input xs, ys, XT may be pd.DataFrame, so change to np.array')
        xs = xs.values
        ys = ys.values
        XT = XT.values
    eval_losses = []
    #oof_predictions = np.zeros((len(xs),))
    test_predictions = np.zeros((len(XT),))
    models = []
    # for idx, (trn_idx, val_idx) in enumerate(folds.split(xs, ys)):
    xs_trn, xs_val, ys_trn, ys_val = train_test_split(xs, ys, test_size=0.25, shuffle=False)

    print(f'now start to train')
    trn_data = lgb.Dataset(xs_trn, label=ys_trn, categorical_feature=category)
    val_data = lgb.Dataset(xs_val, label=ys_val, categorical_feature=category)
    num_round = n_estimators
    clf = lgb.train(param, trn_data, num_round, valid_sets=[
        trn_data, val_data], verbose_eval=verbose, early_stopping_rounds=early_stopping_rounds)
    #oof_predictions[xs_val_idx] = clf.predict(xs_val)
    test_predictions += clf.predict(XT)
    eval_loss = eval_func(ys_val, clf.predict(xs_val))
    print(f'end eval_loss={eval_loss}')
    eval_losses.append(eval_loss)
    models.append(clf)

    return {'eval_loss': np.mean(eval_losses),
            # 'oof_predictions': oof_predictions,
            'test_predictions': test_predictions,
            'models': models}
