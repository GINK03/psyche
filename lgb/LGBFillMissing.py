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
        self.param = None
        self.eval_func = None
        self.fold = None


S = Singleton()


def set_param_template(type):
    if type == 'regression':
        S.param = {
            "objective": "regression_l1",
            "metric": "mae",
            "max_depth": 32,
            "num_leaves": 16,
            "learning_rate": 0.03,
            "bagging_fraction": 0.1,
            "feature_fraction": 0.1,
            "lambda_l1": 0.3,
            "lambda_l2": 0.3,
            "bagging_seed": 777,
            "verbosity": 50,
            "seed": 777,
            # 'max_bin': 512
        }
        S.eval_func = mean_absolute_error

def shot_train(dfTrain, dfTest, target, inplace=False):
    folds = KFold(n_splits=4)
    category = []
    n_estimators = 100000

    feats = set(dfTrain.columns.tolist()) & set(dfTest.columns.tolist())
    dfTrainSlice = dfTrain[feats]
    dfTestSlice = dfTest[feats]
    print(dfTrainSlice.shape)
    print(dfTestSlice.shape)
    dfCon = pd.concat([dfTrainSlice, dfTestSlice], axis=0)
    print(dfCon.shape)
    ys = dfCon[pd.notnull(dfCon[target])][target].values
    xs = dfCon[pd.notnull(dfCon[target])][[c for c in dfCon.columns if c != target]].values
    XT = dfCon[[c for c in dfCon.columns if c != target]].values

    eval_losses = []
    oof_predictions = np.zeros((len(xs),))
    test_predictions = np.zeros((len(XT),))
    models = []
    for idx, (trn_idx, val_idx) in enumerate(folds.split(xs, ys)):
        print(f'now fold={idx:02d} split size is', folds.get_n_splits())
        trn_data = lgb.Dataset(xs[trn_idx], label=ys[trn_idx], categorical_feature=category)
        val_data = lgb.Dataset(xs[val_idx], label=ys[val_idx], categorical_feature=category)
        num_round = n_estimators
        clf = lgb.train(S.param, trn_data, num_round, valid_sets=[
            trn_data, val_data], verbose_eval=150, early_stopping_rounds=10)
        # valのeval_func
        oof_predictions[val_idx] = clf.predict(xs[val_idx])
        test_predictions += clf.predict(XT) / folds.get_n_splits()
        eval_loss = S.eval_func(ys[val_idx], clf.predict(xs[val_idx]))
        print(f'end fold={idx:02d} eval_loss={eval_loss}')
        eval_losses.append(eval_loss)
        models.append(clf)
    dfTrain[f'{target}_fill_missing'] = test_predictions[:len(dfTrain)]
    dfTest[f'{target}_fill_missing'] = test_predictions[len(dfTrain):]

    if inplace:
        for df in [dfTrain, dfTest]:
            df[target] = df[[target, f'{target}_fill_missing']].apply(
                lambda x: x[target] if pd.notnull(x[target]) else x[f'{target}_fill_missing'], axis=1)
            df.drop([f'{target}_fill_missing'], axis=1, inplace=True)
    '''
    return {'eval_loss': np.mean(eval_losses),
            'oof_predictions': oof_predictions,
            'test_predictions': test_predictions,
            'models': models}
    '''
