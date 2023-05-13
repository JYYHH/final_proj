import math
import sklearn
import sklearn.preprocessing
import datetime
import os
import matplotlib.pyplot as plt
from src import *
from time import time
import sys
import seaborn as sns
import pickle
import pandas as pd

# config here

epochs = 100


# first part : plot loss_func

all_train, all_test = pickle.load(open("data/all.pkl", "rb"))
lstm_train, lstm_test = pickle.load(open("data/lstm.pkl", "rb"))
arima_train, arima_test = pickle.load(open("data/arima.pkl", "rb"))
xgboost_train, xgboost_test = pickle.load(open("data/xgboost.pkl", "rb"))
origin_train, origin_test = pickle.load(open("data/origin.pkl", "rb"))
delete_fc_train, delete_fc_test = pickle.load(open("data/delete_fc.pkl", "rb"))

sns.set_theme(style="darkgrid")
df = pd.DataFrame(
        [
            [i + 1 for i in range(epochs)] * 12, 
            ["all_train"] * epochs + ["all_test"] * epochs + \
            ["lstm_train"] * epochs + ["lstm_test"] * epochs + \
            ["arima_train"] * epochs + ["arima_test"] * epochs + \
            ["xgboost_train"] * epochs + ["xgboost_test"] * epochs + \
            ["origin_train"] * epochs + ["origin_test"] * epochs + \
            ["delete_fc_train"] * epochs + ["delete_fc_test"] * epochs \
            , 
            all_train + all_test + \
            lstm_train + lstm_test + \
            arima_train + arima_test + \
            xgboost_train + xgboost_test + \
            origin_train + origin_test + \
            delete_fc_train + delete_fc_test
        ], 
        ["epoch", "type", "Loss"]
    )

sns.lineplot(x = "epoch", y = "Loss",
            hue = "type",
            data=df.T)

plt.show()