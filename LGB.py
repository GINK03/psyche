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


class Singleton(object):
    def __init__(self):
        self.xs = None
        self.ys = None
        self.category = []
        self.eval_func = None
        self.param = None


S = Singleton()


def set_data(xs, ys, category=[], eval_func=None, param=None):
    S.xs = xs
    S.ys = ys
    S.category = category
    if eval_func is not None:
        S.eval_func = eval_func
    else:
        S.eval_func = mean_absolute_error
    if param is not None:
        S.param = param
    else:
        S.param = {
            "objective": "regression_l1",
            "metric": "mae",
            "max_depth": 32,
            "num_leaves": 16,
            "learning_rate": 0.01,
            "bagging_fraction": 0.1,
            "feature_fraction": 0.1,
            "lambda_l1": 0.3,
            "lambda_l2": 0.3,
            "bagging_seed": 777,
            "verbosity": -1,
            "seed": 777,
            # 'max_bin': 512
        }


def train_with_singleton():
    #assert S.xs != None, "must set data first"
    #assert S.ys != None, "must set data first"
    SPLIT_SIZE = 5
    skf = KFold(n_splits=SPLIT_SIZE, shuffle=True,
                random_state=42)
    feature_importance_df = pd.DataFrame()
    eval_losses = []
    models = []
    kf = KFold(n_splits=4)
    predictions = np.zeros((len(S.xs), ))
    for idx, (trn_idx, val_idx) in enumerate(kf.split(S.xs)):
        trn_data = lgb.Dataset(S.xs[trn_idx], label=S.ys[trn_idx])
        val_data = lgb.Dataset(S.xs[val_idx], label=S.ys[val_idx])
        num_round = 10000
        clf = lgb.train(S.param, trn_data, num_round, valid_sets=[
            trn_data, val_data], verbose_eval=100, early_stopping_rounds=10)
        # valのmae
        predictions[val_idx] = clf.predict(S.xs[val_idx])
        eval_loss = S.eval_func(S.ys[val_idx], clf.predict(S.xs[val_idx]))
        print(f'fold={idx} eval_loss={eval_loss}')
        eval_losses.append(eval_loss)
        models.append(clf)
    return {'eval_loss': np.mean(eval_losses),
            'oof_predictions': oof_predictions,
            'test_predictions': test_predictions,
            'models': models}


def shot_train(xs, ys, XT, cats_index, param, fold, eval_func, verbose, early_stopping_rounds, n_estimators):
    if isinstance(xs, (pd.DataFrame)):
        print('input xs, ys, XT may be pd.DataFrame, so change to np.array')
        xs = xs.values
        ys = ys.values
        XT = XT.values
    eval_losses = []
    oof_predictions = np.zeros((len(xs),))
    test_predictions = np.zeros((len(XT),))
    models = []
    for idx, (trn_idx, val_idx) in enumerate(fold.split(xs, ys)):
        print(f'now fold={idx:02d} split size is', fold.get_n_splits())
        trn_data = lgb.Dataset(xs[trn_idx], label=ys[trn_idx], categorical_feature=cats_index)
        val_data = lgb.Dataset(xs[val_idx], label=ys[val_idx], categorical_feature=cats_index)
        num_round = n_estimators
        clf = lgb.train(param, trn_data, num_round, valid_sets=[
            trn_data, val_data], verbose_eval=verbose, early_stopping_rounds=early_stopping_rounds)
        # valのmae
        oof_predictions[val_idx] = clf.predict(xs[val_idx])
        test_predictions += clf.predict(XT) / fold.get_n_splits()
        eval_loss = eval_func(ys[val_idx], clf.predict(xs[val_idx]))
        print(f'end fold={idx:02d} eval_loss={eval_loss}')
        eval_losses.append(eval_loss)
        models.append(clf)

    print(f'total eval loss mean = {np.mean(eval_losses)}')
    return {'eval_loss': np.mean(eval_losses),
            'oof_predictions': oof_predictions,
            'test_predictions': test_predictions,
            'models': models}
