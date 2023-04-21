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
df = df.apply(func = lambda x : x[1:] if x.dtype == object else np.exp(np.diff(np.log(x))) - 1)
date, data = df.date, np.array(df[['CSI1000', 'CSI500', 'CYB', 'HS300', 'SH50']]).astype(np.float64)

a = trans(data)
print(a.shape, a.dtype)

model = GlobalModel()
# model.train(a[:32])
model.train(a[:1000], model.gpu)
# model.test(a[:1000], model.gpu)
model.test(a[1000:], model.gpu)