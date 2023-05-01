from src import *
from time import time
import sys

df = pd.read_csv("index_data.csv")
# date, data = df.date, np.array(df[['CSI1000', 'CSI500', 'CYB', 'HS300', 'SH50']]).astype(np.float64)
for name in ['CSI1000', 'CSI500', 'CYB', 'HS300', 'SH50']:
    date, data = df.date, np.array(df[[name]]).astype(np.float64).reshape(-1)
    # print(data)

    data = np.diff(np.log(data))

    print(data.shape)

    model_set = [
                ARIMA(order = (1,1,0), method='lbfgs', suppress_warnings=True),
                ARIMA(order = (0,1,1), method='lbfgs', suppress_warnings=True),
                ARIMA(order = (1,1,1), method='lbfgs', suppress_warnings=True),
                ARIMA(order = (2,1,0), method='lbfgs', suppress_warnings=True),
                ARIMA(order = (0,1,2), method='lbfgs', suppress_warnings=True),
                ARIMA(order = (1,0,1), method='lbfgs', suppress_warnings=True),
                ARIMA(order = (1,2,1), method='lbfgs', suppress_warnings=True),
                ARIMA(order = (1,1,2), method='lbfgs', suppress_warnings=True),
                ARIMA(order = (2,2,1), method='lbfgs', suppress_warnings=True),
            ]

    # print(f'original_data = {data[20:30]}')
    data = np.array([np.sum(data[i:i+10]) for i in range(0, data.shape[0] - 9, 10)])

    # acc = 0
    # for i in range(100):
    #     can = 0
    #     for idx, model in enumerate(model_set):
    #         model.fit(data[i:i+30])
    #         if model.predict(1) * data[i+30] > 0:
    #             can = 1
    #             break
    #     acc += can
    # print(f"Model {model} predicts acc : {acc / 100}")


    for idx, model in enumerate(model_set):
        for interval in [40, 30, 20, 50, 60, 10]:
            acc = 0
            for i in range(100):
                model.fit(data[i:i+interval])
                acc += np.int32(model.predict(1) * data[i+interval] > 0)
            print(f"Model {model} predicts acc with interval {interval} and on dataset {name} : {acc / 100}")

# print(f"The answer is {data[30:35]}")
