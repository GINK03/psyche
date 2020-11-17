import json
import glob
import sys
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
from sklearn.metrics import roc_auc_score
import EvalFunc


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

def shot_train(X_train, y_train, X_test, category, param, eval_func, verbose, early_stopping_rounds, n_estimators):
    X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=0.25, shuffle=False)

    print(f'now start to train')
    trn_data = lgb.Dataset(X_trn, label=y_trn, categorical_feature=category)
    val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=category)
    num_round = n_estimators
    clf = lgb.train(
            param, 
            trn_data, 
            num_round, 
            valid_sets=[trn_data, val_data], 
            verbose_eval=verbose, 
            early_stopping_rounds=early_stopping_rounds)
    
    test_predictions = np.zeros((len(X_test),))
    test_predictions += clf.predict(X_test)
    eval_loss = eval_func(y_val, clf.predict(X_val))
    print(f'end eval_loss={eval_loss}')

    return {'eval_loss': eval_loss,
            'test_prediction': test_prediction,
            'model': clf}
