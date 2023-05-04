from pmdarima.arima import ARIMA, auto_arima
import torch
import numpy as np
import torch.nn as nn
import time
from xgboost import XGBRegressor
from torch.nn.utils import weight_norm
import torch.nn.functional as F

Index_set = torch.tensor([0, 1, 2, 3, 4])
Index_number = Index_set.shape[0]

class MyARIMA(object):
    # input : (batch_size, index(_quantity), (train_length =) length - 1)
    # output : (batch_size, index(_quantity), oders_length = 16)

    def __init__(self):
        # Index_number index(s)
        
        # hyper_parameters
        self.models = [
            ARIMA(order = (1,1,0), method='lbfgs', suppress_warnings=True),
            ARIMA(order = (0,1,1), method='lbfgs', suppress_warnings=True),
            ARIMA(order = (1,0,1), method='lbfgs', suppress_warnings=True),
            ARIMA(order = (1,1,1), method='lbfgs', suppress_warnings=True),
            ARIMA(order = (2,1,0), method='lbfgs', suppress_warnings=True),
            ARIMA(order = (2,0,1), method='lbfgs', suppress_warnings=True),
            ARIMA(order = (0,1,2), method='lbfgs', suppress_warnings=True),
            ARIMA(order = (0,2,1), method='lbfgs', suppress_warnings=True),
            ARIMA(order = (1,0,2), method='lbfgs', suppress_warnings=True),
            ARIMA(order = (1,2,0), method='lbfgs', suppress_warnings=True),
            ARIMA(order = (1,2,1), method='lbfgs', suppress_warnings=True),
            ARIMA(order = (1,1,2), method='lbfgs', suppress_warnings=True),
            ARIMA(order = (2,1,1), method='lbfgs', suppress_warnings=True),
            ARIMA(order = (2,2,1), method='lbfgs', suppress_warnings=True),
            ARIMA(order = (2,2,2), method='lbfgs', suppress_warnings=True),
        ]
        self.fit_length = 30
        self.onum = 16

        # output of this resemble model
        self.ret = None
        
        self.cnt = 0


    def fit_sub(self, data): # data -> 1-D time series
        # get the correspond data for this frequent
        sub_data = data[-1::-1][:self.fit_length][-1::-1]

        # ret_my
        aic_arr, predict_arr = [], [] # aic && prediction array
        for model in self.models:
            model.fit(sub_data)
            aic_arr.append(model.aic())
            predict_arr.append(model.predict(1))

        # ret_auto
        model = auto_arima(sub_data, start_p=1, start_q=1, start_P=1, start_Q=1,
                 max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=True,
                 stepwise=True, suppress_warnings=True, D=10, max_D=10,
                 error_action='ignore')
        predict_arr.append(model.predict(1))

        self.cnt += 1
        print(f"OK for data : {self.cnt}")

        aic_arr, predict_arr = np.array(aic_arr), np.array(predict_arr)
        return predict_arr


    def fit(self, data):
        batch_size, index_len = data.shape[0], data.shape[1]
        # flatten to 2-D array
        self.ret = np.apply_along_axis(self.fit_sub, 1, data.reshape(-1, data.shape[2]))
        self.ret = self.ret.reshape(batch_size, index_len, len(self.models) + 1)

    def get_ret(self):
        return self.ret

    def get_ret_from_file(self, subset):
        try:
            now_pos, now_ret = 1, np.load("arima_data/1.npy")
            while 1:
                try:
                    now_pos += 1
                    new_matr = np.load(f"arima_data/{now_pos}.npy")
                    now_ret = np.concatenate((now_ret, new_matr), axis=0)
                except:
                    break
                
            return now_ret[subset]
        # (|subset|, index(_quantity), freq_length)
        except:
            return None

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
 
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))

        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
 
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
 
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
 
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

        self.relu = nn.ReLU()
        self.init_weights()
 
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
 
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride = 1, dilation = dilation_size,
                                     padding = (kernel_size-1) * dilation_size, dropout = dropout)]
 
        self.network = nn.Sequential(*layers)
 
    def forward(self, x):
        # x (batch_size, in_channel, length)
        return self.network(x)


class NNModel(torch.nn.Module):
    def __init__(self, onum, hidden_size = 36):
        super(NNModel, self).__init__()
        
        self.arima_len = onum
        self.length = 60 # recently used dates / 60
        self.tcn = TemporalConvNet(5, [5, 5]).double()

        self.fc_between_tcn_lstm = nn.Linear(901, self.length).double()

        self.lstm = nn.LSTM(
                            input_size = Index_number,
                            hidden_size = hidden_size, 
                            num_layers = 2, 
                            batch_first = True,
                            dropout = 0.1,
                            # recurrent_dropout = 0.2
                        ).double() # into float64
        for name, param in self.lstm.named_parameters():
            nn.init.uniform_(param,-0.1,0.1) # initialization

        self.fc = nn.Sequential(
                    nn.Dropout(p = 0.3),
                    # nn.Linear(hidden_size, 20),
                    # nn.LeakyReLU(),
                    nn.Linear(hidden_size, Index_number)
                    # nn.Sigmoid() # 做分类预测，1升/0降
                    # nn.LeakyReLU(),
                    # nn.Linear(200, 100),
                    # nn.LeakyReLU(),
                    # nn.Linear(100, 50),
                    # nn.LeakyReLU(),
                    # nn.Linear(50, Index_number)
                 ).double()

        self.arima_fc = nn.Sequential(
            nn.Linear(self.arima_len * Index_number, self.arima_len * Index_number),
            nn.Tanh(),
            nn.Linear(self.arima_len * Index_number, Index_number),
        ).double()
        self.final_fc = nn.Linear(3 * Index_number, Index_number).double()

    def forward(self, input_seq, arima_data, xgboost_pred):
        # input_seq -> (batch_size, length - 1 = 901, Index_number)
        # arima_data -> (batch_size, orn, Index_number)
        tcn_out = self.tcn(torch.permute(input_seq, (0, 2, 1)))
        # tcn_out -> (batch_size, Index_number, length - 1)
        first_fc_out = nn.Sigmoid()(self.fc_between_tcn_lstm(tcn_out))
        # first_fc_out -> (batch_size, Index_number, self.length)

        lstm_out, _ = self.lstm(torch.permute(first_fc_out, (0, 2, 1))) # only used recently 120 days data
        # lstm_out -> (batch_size, length - 1, hidden_size)
        final_hidden_state = lstm_out[:, -1]
        # final_hidden_state -> (batch_size, hidden_size)
        output_lstm = nn.ReLU()(self.fc(final_hidden_state))
        # output_lstm -> (batch_size, Index_number)
        # return output_lstm

        output_arima = self.arima_fc(arima_data.reshape(arima_data.shape[0], -1))
        # output_arima -> (batch_size, Index_number)
        # return output_arima

        output = self.final_fc(torch.cat((output_lstm, output_arima, xgboost_pred), 1))
        # output -> (batch_size, Index_number)
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
        self.xgb = [
            # 70 %
                    XGBRegressor(n_estimators = 2000, 
                                learning_rate = 0.078588,  
                                max_depth = 6, 
                                early_stopping_rounds = 5
                            ),
            # 66.7 %
                    XGBRegressor(n_estimators = 2000, 
                                learning_rate = 0.005, 
                                max_depth = 6, 
                                early_stopping_rounds = 5
                            ),
            #  71 % 
                    XGBRegressor(n_estimators = 2000, 
                                learning_rate = 0.078588, 
                                max_depth = 6, 
                                early_stopping_rounds = 5
                            ),
            # 53 %
                    XGBRegressor(n_estimators = 2000, 
                                learning_rate = 0.1028, 
                                max_depth = 3, 
                                early_stopping_rounds = 2
                            ),
            # 55.3 %
                    XGBRegressor(n_estimators = 2000, 
                                learning_rate = 0.19972, 
                                max_depth = 4, 
                                early_stopping_rounds = 3
                            )
                   ]
        
        self.nn_model = NNModel(self.arima_model.onum) 
        self.best_nn_model = NNModel(self.arima_model.onum)  
        self.best_metric = 0   

        # hyper_parameters for NN training
        self.lr = 1e-3
        self.wd = 0.2
        self.batch_size = 16
        self.epochs = 20
        self.optimizer_type = "adam"

    def get_model_params(self):
        return self.nn_model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.nn_model.load_state_dict(model_parameters)

    def ret_batch_data(self, data, arimadata):
        data = data.transpose(0, 2, 1)
        arimadata = arimadata.transpose(0, 2, 1)
        # print(data.shape, arimadata.shape)
        # data (data_number, length, index_number)
        # arimadata (data_number, order_number, index_number)
        permu = np.arange(data.shape[0])
        np.random.shuffle(permu)
        return [ (permu[i : i + self.batch_size], data[permu[i : i + self.batch_size], : -1], arimadata[permu[i : i + self.batch_size]], data[permu[i : i + self.batch_size], -1]) for i in range(0, data.shape[0], self.batch_size)]

    def preprocessing(self, data, device):
        # pre_move to GPU && some transforming
        for batch_idx, (sel, x, arima_x, values) in enumerate(data):
            data[batch_idx] = (torch.tensor(sel).to(device), torch.tensor(x).to(device), torch.tensor(arima_x).to(device), torch.tensor(values).to(device))
        Index_set.to(device)

    def xgb_train(self, train_data, dev_data):
        train_data = train_data.transpose(0, 2, 1)
        dev_data = dev_data.transpose(0, 2, 1)
        for which_index in range(5):
            now_xgb = self.xgb[which_index]
            now_xgb.fit(train_data[:, :-1, which_index], train_data[:, -1, which_index], 
                eval_set = [(dev_data[:, :-1, which_index], dev_data[:, -1, which_index])], verbose=False)
            
            # # prediction on dev_dataset         
            # predictions = now_xgb.predict(dev_data[:, :-1, which_index])
            # true_y = dev_data[:, -1, which_index]
            # acc = np.sum(predictions * true_y > 0) / predictions.shape[0]
            # print(f"XGBoost 's acc on dev_set = {acc}")
            
            # # prediction on train_dataset
            # predictions = now_xgb.predict(train_data[:, :-1, which_index])
            # true_y = train_data[:, -1, which_index]
            # acc = np.sum(predictions * true_y > 0) / predictions.shape[0]
            # print(f"XGBoost 's acc on train_set = {acc}")
            # print(predictions[10:20])
            # print(true_y[10:20])

    def xgb_test(self, test_data):
        right, tot = 0, 0
        test_data = test_data.transpose(0, 2, 1)
        for which_index in Index_set:
            now_xgb = self.xgb[which_index]
            predictions = now_xgb.predict(test_data[:, :-1, which_index])
            true_y = test_data[:, -1, which_index]
            tot += predictions.shape[0]
            # print(predictions[30:50], true_y[30:50])
            right += np.sum(predictions * true_y > 0)
        
        print(f"XGB test : Acc on test data is {right / tot}")

    def xgb_predict(self, infer_data):
        # data -> (batch_size, Index_number, length)
        ret = []
        for idx, which_index in enumerate(Index_set):
            now_xgb = self.xgb[which_index]
            now_data = infer_data[:, idx]
            # print(now_data.shape)
            predictions = now_xgb.predict(now_data)
            ret.append(predictions)
        # ret -> (batch_size, Index_number)
        ret = np.array(ret, dtype = np.float64).T
        ret = torch.tensor(ret).to(self.gpu)
        return ret

    def train(self, data, xgb_data, train_index, dev_index, device = "cpu"):
        self.nn_model.to_device(device)
        self.best_nn_model.to_device(device)
        self.nn_model.train()
        train_data_raw, dev_data_raw = data[train_index], data[dev_index]

        self.xgb_train(xgb_data[train_index], xgb_data[dev_index])
        # self.xgb_test(xgb_data[dev_index])
        # exit(0)

        # train and update
        criterion = nn.MSELoss().to(device) # Regression
        # criterion = nn.BCELoss().to(device) # Classification

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
                            amsgrad=True,
                            betas = (0.9, 0.999)
                        )

        begin_time = time.time()

        epoch_loss = []
        for epoch in range(self.epochs):
            # transform the data -------> now combine the output of ARIMA Models
            train_data = self.ret_batch_data(train_data_raw, self.arima_model.get_ret_from_file(train_index))
            dev_data = self.ret_batch_data(dev_data_raw, self.arima_model.get_ret_from_file(dev_index))
            # move to device(GPU)
            self.preprocessing(train_data, device)
            self.preprocessing(dev_data, device)
            
            batch_loss, tot_sample = [], 0
            for batch_idx, (sel, x, arima_x, values) in enumerate(train_data):
                # x, values = x.to(device), values.to(device)
                self.nn_model.zero_grad()
                Pred = self.nn_model(
                    x[:, :, Index_set], 
                    arima_x[:, :, Index_set], 
                    self.xgb_predict(
                        xgb_data[train_index][sel.cpu().numpy()][:, Index_set, :-1]
                    )
                ).reshape(-1)
                # print(Pred.shape, values.reshape(x.shape[0], -1)[:, Index_set].reshape(-1).shape)
                loss = criterion(Pred, values.reshape(x.shape[0], -1)[:, Index_set].reshape(-1)) * x.shape[0]
                loss.backward()

                optimizer.step()
                batch_loss.append(loss.item() * x.shape[0])
                tot_sample += x.shape[0]

            epoch_loss.append(sum(batch_loss) / tot_sample)
            print('Epoch: {} \nAverage data point Loss(MSE): {:.6f}'.format(epoch, epoch_loss[-1]))

            # use the dev-data-set to test the model in independent dataset
            with torch.no_grad():
                tot, right = 0, 0
                for batch_idx, (sel, x, arima_x, values) in enumerate(dev_data):
                    # x, values = x.to(device), values.to(device)
                    pred = self.nn_model(
                        x[:, :, Index_set], 
                        arima_x[:, :, Index_set], 
                        self.xgb_predict(
                            xgb_data[dev_index][sel.cpu().numpy()][:, Index_set, :-1]
                        )
                    ).reshape(-1)
                    loss = criterion(pred, values.reshape(x.shape[0], -1)[:, Index_set].reshape(-1)) * x.shape[0]

                    # _, predicted = torch.max(pred, -1)
                    tot += x.shape[0] * Index_number
                    right += torch.sum((pred * values.reshape(x.shape[0], -1)[:, Index_set].reshape(-1)) > 0)

                    if batch_idx <= 2:
                        # print(x[:3, :, Index_set].reshape(3, -1))
                        print(pred[:3], values.reshape(x.shape[0], -1)[:, Index_set].reshape(-1)[:3], end = '\n')
                acc = right/tot
                print('ACC in dev dataset: {:.6f}'.format(acc))

                # only save the best model !
                if acc >= self.best_metric :
                    self.best_metric = acc
                    self.best_nn_model.load_state_dict(self.nn_model.state_dict())
        
        print(f'total time = {time.time() - begin_time} s')
        print(f'Best acc in valid set : {self.best_metric}')

        print(self.best_nn_model)
        # print(self.best_nn_model.state_dict())

    def test(self, data, xgb_data, test_index, device = "cpu"):
        self.best_nn_model.to(device)
        self.best_nn_model.eval()

        # print(self.best_nn_model.state_dict())

        test_data = data[test_index]

        metrics = {
            'total_loss': 0.0,
            'right_number': 0,
            'test_total': 0
        }

        criterion = nn.MSELoss().to(device) # Regression
        # criterion = nn.BCELoss().to(device) # Classification
        # transform the data
        test_data = self.ret_batch_data(test_data, self.arima_model.get_ret_from_file(test_index))
        # pre_move to GPU
        self.preprocessing(test_data, device)

        # NNN = None
        with torch.no_grad():
            for batch_idx, (sel, x, arima_x, values) in enumerate(test_data):
                # NNN = arima_x
                pred = self.nn_model(
                    x[:, :, Index_set], 
                    arima_x[:, :, Index_set], 
                    self.xgb_predict(
                        xgb_data[test_index][sel.cpu().numpy()][:, Index_set, :-1]
                    )
                ).reshape(-1)
                # print(pred.shape, values.reshape(x.shape[0], -1)[:, Index_set].reshape(-1).shape)
                loss = criterion(pred, values.reshape(x.shape[0], -1)[:, Index_set].reshape(-1)) * x.shape[0]

                metrics['total_loss'] += loss.item() * x.shape[0]
                metrics['test_total'] += x.shape[0] * Index_number
                metrics['right_number'] += torch.sum((pred * values.reshape(x.shape[0], -1)[:, Index_set].reshape(-1)) > 0)

        print('ACC = {}'.format(metrics['right_number'] / metrics['test_total']))

        # x = torch.normal(0, 1, (3, 61, 1)).to(self.gpu).double()
        # print(x[0], x[1], x[2])
        # print(self.best_nn_model(x[:1], NNN))
        # print(self.best_nn_model(x[1:2], NNN))
        # print(self.best_nn_model(x[2:3], NNN))
        # print(self.best_nn_model(test_data[0][0][:, :, Index_set], test_data[0][1][:, :, Index_set]))
        # print(self.best_nn_model(test_data[1][0][:, :, Index_set], test_data[1][1][:, :, Index_set]))

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