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
import EvalFunc
class Singleton(object):
    def __init__(self):
        self.xs = None
        self.ys = None
S = Singleton()
def set_data(xs,ys):
    S.xs=xs
    S.ys=ys

def trainer(max_depth, num_leaves, bagging_fraction, feature_fraction, lambda_l1, lambda_l2):
    param = {
        "objective": "regression_l1",
        "metric": "mae",
        "max_depth": max_depth,
        "num_leaves": num_leaves,
        "learning_rate": 0.01,
        "bagging_fraction": bagging_fraction,
        "feature_fraction": feature_fraction,
        "lambda_l1": lambda_l1,
        "lambda_l2": lambda_l2,
        "bagging_seed": 777,
        "verbosity": -1,
        "seed": 777,
        # 'max_bin': 512
    }
    feature_importance_df = pd.DataFrame()

    eval_losses = []
    kf = KFold(n_splits=4)
    predictions = np.zeros((len(S.xs), ))
    models = []
    for idx, (trn_idx, val_idx) in enumerate(kf.split(S.xs)):
        trn_data = lgb.Dataset(S.xs[trn_idx], label=S.ys[trn_idx])
        val_data = lgb.Dataset(S.xs[val_idx], label=S.ys[val_idx])
        num_round = 10000
        clf = lgb.train(param, trn_data, num_round, valid_sets=[
            trn_data, val_data], verbose_eval=0, early_stopping_rounds=10)
        predictions[val_idx] = clf.predict(S.xs[val_idx])
        models.append(clf)
        #mae = metrics.mean_absolute_error(ys[val_idx], clf.predict(xs[val_idx]))
        mape = EvalFunc.mape(S.ys[val_idx], clf.predict(S.xs[val_idx]))
        print(idx, mape)
        eval_losses.append(mape)

    if finish is False:
        return np.mean(eval_losses)
    else:
        return (np.mean(eval_losses), models)

def objective(trial):
    max_depth = trial.suggest_int('max_depth', 2, 100)
    num_leaves = trial.suggest_int('num_leaves', 2, 100)
    bagging_fraction = trial.suggest_uniform('bagging_fraction', 0, 1)
    feature_fraction = trial.suggest_uniform('feature_fraction', 0, 1)
    lambda_l1 = trial.suggest_uniform('lambda_l1', 0, 5)
    lambda_l2 = trial.suggest_uniform('lambda_l2', 0, 5)
    return trainer(max_depth, num_leaves, bagging_fraction, feature_fraction, lambda_l1, lambda_l2)

finish = False
def run(n_trials=10):
    global finish
    study = optuna.create_study()
    study.optimize(objective, n_trials=n_trials)
    best_param = study.best_params
    print('train with best param and dump features.')
    finish = True
    eval_loss, models = trainer(**best_param)
    print('eval_loss', eval_loss)
    return models
