from src import *
from time import time
import sys

begin_index, end_index, id_ = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

df = pd.read_csv("index_data.csv")
df = df.apply(func = lambda x : x[1:] if x.dtype == object else np.diff(np.log(x)))
date, data = df.date, np.array(df[['CSI1000', 'CSI500', 'CYB', 'HS300', 'SH50']])

a = trans(raw_data = data[begin_index : end_index + 930], length = 931, interval = 30, after = 30)

model = MyARIMA()

BE = time()
model.fit(a[:, :, :-1]) # features
print(f'Used time : {time() - BE}')
print(a[:, :, -1]) # labels

np.save(f'arima_data/{id_}.npy', model.ret)
