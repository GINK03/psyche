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

def trainer():
    #assert S.xs != None, "must set data first"
    #assert S.ys != None, "must set data first"
    SPLIT_SIZE = 5
    skf = KFold(n_splits=SPLIT_SIZE, shuffle=True,
                random_state=42)
    param = {
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
    feature_importance_df = pd.DataFrame()
    maes = []
    kf = KFold(n_splits=4)
    predictions = np.zeros((len(S.xs), ))
    for idx, (trn_idx, val_idx) in enumerate(kf.split(S.xs)):

        trn_data = lgb.Dataset(S.xs[trn_idx], label=S.ys[trn_idx])
        val_data = lgb.Dataset(S.xs[val_idx], label=S.ys[val_idx])
        num_round = 10000
        clf = lgb.train(param, trn_data, num_round, valid_sets=[
            trn_data, val_data], verbose_eval=100, early_stopping_rounds=10)
        # valのmae
        predictions[val_idx] = clf.predict(S.xs[val_idx])
        mape=EvalFunc.mape(S.ys[val_idx], clf.predict(S.xs[val_idx]))
        print(mape)
        #mae = metrics.mean_absolute_error(S.ys[val_idx], clf.predict(S.xs[val_idx]))
        #print(mae)
        #maes.append(mae)
