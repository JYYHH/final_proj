import math
import sklearn
import sklearn.preprocessing
import datetime
import os
import matplotlib.pyplot as plt
from src import *
from time import time

df = pd.read_csv("index_data.csv")
df = df.apply(func = lambda x : x[1:] if x.dtype == object else np.diff(np.log(x)))
date, data = df.date, np.array(df[['CSI1000', 'CSI500', 'CYB', 'HS300', 'SH50']])

a = trans(data)
print(a.shape)

model = MyARIMA()

BE = time()
model.fit(a[:4, :, :-1])
print(f'Used time : {time() - BE}')

print(model.ret, a[:4, :, -1])
np.save('data/test.npy', model.ret)
