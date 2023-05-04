import math
import sklearn
import sklearn.preprocessing
import datetime
import os
import matplotlib.pyplot as plt
from src import *
from time import time
import sys

df = pd.read_csv("index_data.csv")
df = df.apply(func = lambda x : x[1:] if x.dtype == object else np.diff(np.log(x)))
date, data = df.date, np.array(df[['CSI1000', 'CSI500', 'CYB', 'HS300', 'SH50']]).astype(np.float64)

ALL, part = trans(raw_data = data, length = 931, interval = 1, after = 30), trans(raw_data = data, length = 931, interval = 15, after = 30)
print(ALL.shape, ALL.dtype)
print(part.shape, part.dtype)

model = GlobalModel()
# model.train(a[:32])
model.train(ALL, part, np.arange(1000), np.arange(1000, 1300), model.gpu)
# model.test(a[:1000], model.gpu)
model.test(ALL, part, np.arange(1200, ALL.shape[0]), model.gpu)
model.test(ALL, part, np.arange(1000), model.gpu)
model.test(ALL, part, np.arange(1000, 1300), model.gpu)

