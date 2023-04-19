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
date, data = df.date, np.array(df[['CSI1000', 'CSI500', 'CYB', 'HS300', 'SH50']])

a = trans(data)
print(a.shape)

# model = MyARIMA()

# BE = time()
# model.fit(a[:, :, :-1]) # features
# print(f'Used time : {time() - BE}')
# print(model.ret, a[:, :, -1]) # labels

# np.save('data/test.npy', model.ret)
