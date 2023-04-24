from pmdarima.arima import ARIMA, auto_arima
import torch
import numpy as np
import torch.nn as nn
import time

class MyARIMA(object):
    # input : (batch_size, index(_quantity), (train_length =) length - 1)
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

    def get_ret_from_file(self, l, r):
        try:
            now_pos, now_ret = 1, np.load("arima_data/1.npy")
            while 1:
                try:
                    now_pos += 1
                    new_matr = np.load(f"arima_data/{now_pos}.npy")
                    now_ret = np.concatenate((now_ret, new_matr), axis=0)
                except:
                    break
                
            return now_ret[l : r]
        except:
            return None

class NNModel(torch.nn.Module):
    def __init__(self, freq, hidden_size = 32):
        super(NNModel, self).__init__()
        
        self.freq = freq
        self.length = 121 # recently used dates

        self.lstm = nn.LSTM(
                            input_size = 5,
                            hidden_size = hidden_size, 
                            num_layers=2, 
                            batch_first=True
                        ).double() # into float64
        self.fc = nn.Linear(hidden_size, 5).double()
        self.arima_fc = nn.Linear(freq * 5, 5).double()
        self.final_fc = nn.Linear(10, 5).double()

    def forward(self, input_seq, arima_data):
        # input_seq -> (batch_size, length - 1, 5)
        # arima_data -> (batch_size, freq, 5)
        lstm_out, _ = self.lstm(input_seq[:, -self.length:]) # only used recently 120 days data
        # lstm_out -> (batch_size, length - 1, hidden_size)
        final_hidden_state = lstm_out[:, -1]
        # final_hidden_state -> (batch_size, hidden_size)
        output_lstm = nn.Tanh()(self.fc(final_hidden_state))
        # output_lstm -> (batch_size, 5)
        output_arima = nn.Tanh()(self.arima_fc(arima_data.reshape(arima_data.shape[0], -1)))
        # output_arima -> (batch_size, 5)

        output = self.final_fc(torch.cat((output_lstm, output_arima), 1))
        # output -> (batch_size, 5)
        return output

    def to_device(self, device):
        self.to(device)

class GlobalModel(object):     
    # input : (data_enrty_number, index(_quantity), length)
    # mid_input : (batch_size, freq_length(12), fit_length(30), index) / (batch_size, index)

    def __init__(self):
        self.cpu = "cpu"
        self.gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # models
        self.arima_model = MyARIMA()
        self.nn_model = NNModel(len(self.arima_model.freq))        

        # hyper_parameters for NN training
        self.lr = 1e-4
        self.wd = 0.0
        self.batch_size = 4
        self.epochs = 100
        self.optimizer_type = "sgd"

    def get_model_params(self):
        return self.nn_model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.nn_model.load_state_dict(model_parameters)

    def ret_batch_data(self, data, arimadata):
        print(data.shape, arimadata.shape)
        data = data.transpose(0, 2, 1)
        arimadata = arimadata.transpose(0, 2, 1)
        return [ (data[i : i + self.batch_size, : -1], arimadata[i : i + self.batch_size], data[i : i + self.batch_size, -1]) for i in range(0, data.shape[0], self.batch_size)]

    def preprocessing(self, data, device):
        # pre_move to GPU && some transforming
        for batch_idx, (x, arima_x, values) in enumerate(data):
            data[batch_idx] = (torch.tensor(x).to(device), torch.tensor(arima_x).to(device), torch.tensor(values).to(device))
            

    def train(self, train_data, device = "cpu"):
        self.nn_model.to_device(device)
        self.nn_model.train()

        # train and update
        criterion = nn.MSELoss().to(device)

        if self.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                            filter(lambda p: p.requires_grad, self.nn_model.parameters()),
                            lr = self.lr, 
                            momentum = 0.9, 
                            weight_decay = 0, 
                            dampening = 0, 
                            nesterov = "false"
                        )
        else:
            optimizer = torch.optim.Adam(
                            filter(lambda p: p.requires_grad, self.nn_model.parameters()), 
                            lr = self.lr,
                            weight_decay = self.wd, 
                            amsgrad=True
                        )
        
        # transform the data -------> now combine the output of ARIMA Models
        self.train_len = train_data.shape[0]
        train_data = self.ret_batch_data(train_data, self.arima_model.get_ret_from_file(0, self.train_len))
        # preprocessing   <----------> x : (batch_size, length-1, 5) -----> # x : (batch, freq, fit_length, 5)
        # and move to device(GPU)
        self.preprocessing(train_data, device)

        begin_time = time.time()

        epoch_loss = []
        for epoch in range(self.epochs):
            batch_loss, tot_sample = [], 0
            for batch_idx, (x, arima_x, values) in enumerate(train_data):
                # x, values = x.to(device), values.to(device)
                self.nn_model.zero_grad()
                Pred = self.nn_model(x, arima_x).reshape(-1)
                loss = criterion(Pred, values.reshape(-1)) * x.shape[0]
                loss.backward()

                optimizer.step()
                batch_loss.append(loss.item() * x.shape[0])
                tot_sample += x.shape[0]

            epoch_loss.append(sum(batch_loss) / tot_sample)
            print('Epoch: {} \nAverage data point Loss(MSE): {:.6f}'.format(epoch, epoch_loss[-1]))
        
        print(f'total time = {time.time() - begin_time} s')

    def test(self, test_data, device = "cpu"):
        self.nn_model.to(device)
        self.nn_model.eval()

        metrics = {
            'total_loss': 0.0,
            'right_number': 0,
            'test_total': 0
        }

        criterion = nn.MSELoss().to(device)

        # transform the data
        test_data = self.ret_batch_data(test_data, self.arima_model.get_ret_from_file(self.train_len, 6666))
        # pre_move to GPU
        self.preprocessing(test_data, device)

        with torch.no_grad():
            for batch_idx, (x, arima_x, values) in enumerate(test_data):
                # x, values = x.to(device), values.to(device)
                pred = self.nn_model(x, arima_x).reshape(-1)
                loss = criterion(pred, values.reshape(-1))

                # _, predicted = torch.max(pred, -1)
                metrics['total_loss'] += loss.item() * x.shape[0]
                metrics['test_total'] += x.shape[0]
                metrics['right_number'] += torch.sum((pred * values.reshape(-1)) > 0)

        print('ACC = {}'.format(metrics['right_number'] / (metrics['test_total'] * 5)))

        return metrics







########################################################################################################
    # __init__
        # self.lstms = [ nn.LSTM(
        #                     input_size = 5,
        #                     hidden_size = hidden_size, 
        #                     num_layers=2, 
        #                     batch_first=True
        #                 ).double() # into float64
        #               for i in range(freq)
        #             ]   

        # self.fcs = [ nn.Linear(hidden_size, 5).double() for i in range(freq) ] # into float64

        # self.last_MLP = nn.Sequential(
        #                     nn.Tanh(),
        #                     # nn.Linear(5 * self.freq, hidden_size * 5),
        #                     # nn.ReLU(),
        #                     # nn.Linear(hidden_size * 5, hidden_size),
        #                     # nn.ReLU(),
        #                     # nn.Linear(hidden_size, 5)
        #                     nn.Linear(5 * self.freq, 5)
        #                 ).double() # into float64

    # forward
        # # # input_seq -> (batch_size, freq_length(12), fit_length(30), 5)
        # mid = [ self.fcs[i](self.lstms[i](input_seq[:, i])[0][:, -1]).reshape(1, -1, 5) for i in range(self.freq) ]
        # # mid = [(1, batch_size, 5) * freq ]'s list
        # mid2 = torch.permute(torch.cat(mid, 0), (1, 0, 2))
        # # mid2 = (batch_size, freq, 5)
        # output = self.last_MLP(mid2.reshape(mid2.shape[0], -1))
        # # output = (batch_size, 5)
        # return output

    # to_device
        # for lstm in self.lstms:
        #     lstm.to(device)
        # for fc in self.fcs:
        #     fc.to(device)
        # self.last_MLP.to(device)

    # preprocessing
            # # x : (batch_size, length-1, 5)
            # x = torch.tensor(np.array([ x[:, -f::-f][:, :self.arima_model.fit_length][:, -1::-1] for f in self.arima_model.freq ])).double()
            # # x : (freq, batch, fit_length, 5)
            # x = torch.permute(x, (1, 0, 2, 3))
            # # x : (batch, freq, fit_length, 5)
            # data[batch_idx] = (x.to(device), torch.tensor(values).to(device))