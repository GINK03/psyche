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
from sklearn import linear_model


def shot_train(xs, ys, XT, folds, eval_func):
    if isinstance(xs, (pd.DataFrame)):
        print('input xs, ys, XT may be pd.DataFrame, so change to np.array')
        xs = xs.fillna(0.0).replace([np.inf, -np.inf], 0).values
        ys = ys.fillna(0.0).replace([np.inf, -np.inf], 0).values
        XT = XT.fillna(0.0).replace([np.inf, -np.inf], 0).values
    eval_losses = []
    oof_predictions = np.zeros((len(xs),))
    test_predictions = np.zeros((len(XT),))
    models = []
    for idx, (trn_idx, val_idx) in enumerate(folds.split(xs)):
        print(f'now fold={idx:02d} split size is', folds.get_n_splits())
        trn_xs, trn_ys = xs[trn_idx], ys[trn_idx]
        #val_xs, val_ys = xs[val_idx], ys[val_idx]
        clf = linear_model.Lasso(alpha=0.1)
        clf.fit(trn_xs, trn_ys)
        oof_predictions[val_idx] = clf.predict(xs[val_idx])
        test_predictions += clf.predict(XT) / folds.get_n_splits()
        eval_loss = eval_func(ys[val_idx], clf.predict(xs[val_idx]))
        print(f'end fold={idx:02d} eval_loss={eval_loss}')
        eval_losses.append(eval_loss)
        models.append(clf)

    return {'eval_loss': np.mean(eval_losses),
            'oof_predictions': oof_predictions,
            'test_predictions': test_predictions,
            'models': models}
