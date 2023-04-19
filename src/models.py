from pmdarima.arima import ARIMA, auto_arima
import torch
import numpy as np

class MyARIMA(object):
    # input : (batch_size, index(_quantity), (train_)length)
    # output : (batch_size, index(_quantity), freq_length)

    def __init__(self):
        # 5 index(s)
        
        # hyper_parameters
        self.models = [
            ARIMA(order = (1,1,0), method='lbfgs', suppress_warnings=True),
            ARIMA(order = (0,1,1), method='lbfgs', suppress_warnings=True),
            ARIMA(order = (1,1,1), method='lbfgs', suppress_warnings=True),
            ARIMA(order = (2,1,1), method='lbfgs', suppress_warnings=True),
            ARIMA(order = (2,1,0), method='lbfgs', suppress_warnings=True)
        ]
        self.fit_length = 30
        self.freq = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

        # output of this resemble model
        self.ret = None
        
        self.cnt = 0
    
    # hyper_param
    def prob_func(self, Y):
        # soft_max
        # mid = np.exp(Y.min() - Y)
        # return mid / mid.sum()

        # 1/x
        mid = 1.0 / (Y - Y.min() + 1)
        return mid / mid.sum()


    def fit_sub(self, data): # data -> 1-D time series
        ret_my, ret_auto = [], []

        for f in self.freq:
            print(f'Now freq = {f}')
            # get the correspond data for this frequent
            sub_data = data[-f::-f][:self.fit_length][-1::-1]

            # ret_my
            aic_arr, predict_arr = [], [] # aic && prediction array
            for model in self.models:
                model.fit(sub_data)
                aic_arr.append(model.aic())
                predict_arr.append(model.predict(1))
            aic_arr, predict_arr = np.array(aic_arr), np.array(predict_arr)
            # print(aic_arr, predict_arr)
            ret_my.append(np.float64(np.dot(predict_arr.reshape(-1), self.prob_func(aic_arr).reshape(-1))))

            # ret_auto
            # model = auto_arima(sub_data, start_p=1, start_q=1, start_P=1, start_Q=1,
            #          max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=True,
            #          stepwise=True, suppress_warnings=True, D=10, max_D=10,
            #          error_action='ignore')
            # ret_auto.append(np.float64(model.predict(1)))

        self.cnt += 1
        print(f"OK for data : {self.cnt}")

        return np.array(ret_my + ret_auto)


    def fit(self, data):
        batch_size, index_len = data.shape[0], data.shape[1]
        # flatten to 2-D array
        self.ret = np.apply_along_axis(self.fit_sub, 1, data.reshape(-1, data.shape[2]))
        self.ret = self.ret.reshape(batch_size, index_len, -1)

    def get_ret(self):
        return self.ret


class NNModel(torch.nn.Module):
    def __init__(self):
        self.cpu = "cpu"
        self.gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
