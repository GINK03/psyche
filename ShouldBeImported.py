# here is for CV
# import torch
# import torchvision
import cv2
#import PytorchUnet
#import PytorchDataloaderUnet
#from catboost import CatBoostRegressor

# Generics Purpose
from collections import Counter
from collections import namedtuple

# here is for tablue
from IPython.core.display import display, HTML
import AdversarielClassify
import GetFeatureImportance
import GetCorHeatMap
import GetCor
import CountEncoder
import LabelEncoder
import ExpectIsCategorical
import EvalFunc
import LGBHoldout
import LGBSearch
import LGBFillMissing
import LGB
import XGB
import Lasso
import Ser
import ReduceMemory
import PCA
import PercentailClipper
import TargetEncoder
import Rank
import pickle
import networkx as nx
import altair as alt
import json
from IPython.display import HTML
import warnings
import gc
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
#from sklearn.preprocessing import LabelEncoder
import datetime
import time
import xgboost as xgb
import lightgbm as lgb
from importlib import reload
import pandas_profiling as pdp
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR, SVR
from sklearn.metrics import mean_absolute_error

# 描画関連
pd.options.display.precision = 15
warnings.filterwarnings("ignore")
# COL数
pd.set_option("display.max_columns", None)
# ROW数
pd.set_option('display.max_rows', 1000)
# 幅
display(HTML("<style>.container { width:100% !important; }</style>"))

# 最後の出力
print('All components loads was done! Ready!')
