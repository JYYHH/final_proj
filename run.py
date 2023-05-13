import math
import sklearn
import sklearn.preprocessing
import datetime
import os
import matplotlib.pyplot as plt
from src import *
from time import time
import sys
import pickle

df = pd.read_csv("index_data.csv")
df = df.apply(func = lambda x : x[1:] if x.dtype == object else np.diff(np.log(x)))
date, data = df.date, np.array(df[['CSI1000', 'CSI500', 'CYB', 'HS300', 'SH50']]).astype(np.float64)

ALL, part = trans(raw_data = data, length = 931, interval = 1, after = 30), trans(raw_data = data, length = 931, interval = 15, after = 30)
print(ALL.shape, ALL.dtype)
print(part.shape, part.dtype)

try:
    model = GlobalModel(sys.argv[1], [int(sys.argv[2])])
    _, train_loss_list, dev_loss_list = model.train(
        data = ALL, 
        xgb_data = part, 
        train_index = np.arange(1000), 
        dev_index = np.arange(1000, 1300), 
        device = model.gpu,
        epochs = 100,
        learning_rate = 3 * 1e-3,
        loss_func = "MSE"
    )
    train_dataset_metrics = model.test(ALL, part, np.arange(1000), model.gpu)
    test_dataset_metrics = model.test(ALL, part, np.arange(1000, 1300), model.gpu)
    print(f"Train Metric : {train_dataset_metrics}")
    print("------------------------------------------------------")
    print(f"Test Metric : {test_dataset_metrics}")
except:
    model = GlobalModel(sys.argv[1], [0, 1, 2, 3, 4])
    _, train_loss_list, dev_loss_list = model.train(
            data = ALL, 
            xgb_data = part, 
            train_index = np.arange(1000), 
            dev_index = np.arange(1000, 1300), 
            device = model.more_gpu,
            epochs = 100,
            learning_rate = 3 * 1e-3,
            loss_func = "MSE"
        )
    train_dataset_metrics = model.test(ALL, part, np.arange(1000), model.more_gpu)
    test_dataset_metrics = model.test(ALL, part, np.arange(1000, 1300), model.more_gpu)
    print(f"Train Metric : {train_dataset_metrics}")
    print("------------------------------------------------------")
    print(f"Test Metric : {test_dataset_metrics}")
    pickle.dump((train_loss_list, dev_loss_list), open(sys.argv[2], "wb"))

