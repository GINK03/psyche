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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ShuffleSplit
import numpy as np

def adversariel_train(xs, XT, category):
    if isinstance(xs, (pd.DataFrame)):
        print('input xs, ys, XT may be pd.DataFrame, so change to np.array')
        xs = xs.values
        XT = XT.values
    ys = np.array([1]*len(xs) + [0]*len(XT))
    xs = np.vstack([xs, XT])

    folds = ShuffleSplit(n_splits=4, random_state=11)
    eval_func = roc_auc_score
    verbose = 50
    early_stopping_rounds = 50
    n_estimators = 5000
    param = {'learning_rate': 0.001, 'max_depth': 100, 'num_leaves': 60, 'bagging_fraction':1.0, 
             'feature_fraction': 1.0, 'lambda_l1': 0., 'lambda_l2': 0.}
    param['objective'] = 'binary'
    param['metric'] = 'auc'

    eval_losses = []
    oof_predictions = np.zeros((len(xs),))
    models = []
    for idx, (trn_idx, val_idx) in enumerate(folds.split(xs)):
        print(f'now fold={idx:02d} split size is', folds.get_n_splits())
        trn_data = lgb.Dataset(xs[trn_idx], label=ys[trn_idx], categorical_feature=category)
        val_data = lgb.Dataset(xs[val_idx], label=ys[val_idx], categorical_feature=category)
        num_round = n_estimators
        clf = lgb.train(param, trn_data, num_round, valid_sets=[
            trn_data, val_data], verbose_eval=verbose, early_stopping_rounds=early_stopping_rounds)
        # valのmae
        oof_predictions[val_idx] = clf.predict(xs[val_idx])
        eval_loss = roc_auc_score(ys[val_idx].flatten(), clf.predict(xs[val_idx]).flatten())
        print(f'end fold={idx:02d} eval_loss={eval_loss}')
        eval_losses.append(eval_loss)
        models.append(clf)

    return {'eval_loss': np.mean(eval_losses),
            'oof_predictions': oof_predictions,
            'models': models}
