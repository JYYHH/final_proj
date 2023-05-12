from pmdarima.arima import ARIMA, auto_arima
import torch
import numpy as np
import torch.nn as nn
import time
from xgboost import XGBRegressor
from torch.nn.utils import weight_norm
import torch.nn.functional as F

Print_XGboost = 0

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
    def __init__(self, onum, TYPE = "all", hidden_size = 36, index_number = 5):
        super(NNModel, self).__init__()
        
        self.arima_len = onum
        self.length = 60 # recently used dates / 60
        self.tcn = TemporalConvNet(5, [5, 5]).double()
        self.TYPE = TYPE
        self.Index_number = index_number

        self.fc_between_tcn_lstm = nn.Linear(901, self.length).double()

        self.lstm = nn.LSTM(
                            input_size = self.Index_number,
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
                    nn.Linear(hidden_size, self.Index_number)
                    # nn.Sigmoid() # 做分类预测，1升/0降
                    # nn.LeakyReLU(),
                    # nn.Linear(200, 100),
                    # nn.LeakyReLU(),
                    # nn.Linear(100, 50),
                    # nn.LeakyReLU(),
                    # nn.Linear(50, Index_number)
                 ).double()

        self.arima_fc = nn.Sequential(
            nn.Linear(self.arima_len * self.Index_number, self.arima_len * self.Index_number),
            nn.Tanh(),
            nn.Linear(self.arima_len * self.Index_number, self.Index_number),
        ).double()
        self.final_fc = nn.Linear(3 * self.Index_number, self.Index_number).double()

    def forward(self, input_seq, arima_data, xgboost_pred):
        # input_seq -> (batch_size, length = 901, Index_number)
        # arima_data -> (batch_size, orn, Index_number)
        tcn_out = self.tcn(torch.permute(input_seq, (0, 2, 1)))
        # tcn_out -> (batch_size, Index_number, length = 901)
        # first_fc_out = nn.Sigmoid()(self.fc_between_tcn_lstm(tcn_out)) # fc
        # first_fc_out -> (batch_size, Index_number, self.length)

        if self.type == "all":
            # transfer to 15 days interval
            tcn_out = tcn_out[:, :, 1:]
            sum_up_ = tcn_out.reshape(tcn_out.shape[0], tcn_out.shape[1], -1, 15)
            sum_up_ = sum_up_.sum(axis = 3, keepdims = False)
                # sum_up_ -> (batch_size, Index_number, self.length)

            lstm_out, _ = self.lstm(torch.permute(sum_up_, (0, 2, 1))) # only used recently 120 days data
        elif self.type == "delete_fc":
            lstm_out, _ = self.lstm(torch.permute(tcn_out[:, :, -self.length:], (0, 2, 1)))
        else:
            first_fc_out = nn.Sigmoid()(self.fc_between_tcn_lstm(tcn_out)) # fc
            lstm_out, _ = self.lstm(torch.permute(first_fc_out, (0, 2, 1)))


        # lstm_out, _ = self.lstm(input_seq[:, -self.length:])
        # lstm_out -> (batch_size, self.length, hidden_size)
        final_hidden_state = lstm_out[:, -1]
        # final_hidden_state -> (batch_size, hidden_size)
        output_lstm = self.fc(final_hidden_state)
        # output_lstm -> (batch_size, Index_number)
        # return output_lstm

        output_arima = self.arima_fc(arima_data.reshape(arima_data.shape[0], -1))
        # output_arima -> (batch_size, Index_number)
        # return output_arima

        output = self.final_fc(torch.cat((output_lstm, output_arima, xgboost_pred), 1))
        # output -> (batch_size, Index_number)

        if self.TYPE == "all" or self.TYPE == "origin" or self.TYPE == "delete_fc":
            return output
        elif self.TYPE == "lstm":
            return output_lstm
        elif self.TYPE == "arima":
            return output_arima
        elif self.TYPE == "xgboost":
            return xgboost_pred
        else:
            print("Wrong TYPE of Model")
            exit(9)


    def to_device(self, device):
        self.to(device)

class GlobalModel(object):     
    # input : (data_enrty_number, index(_quantity), length)
    # mid_input : (batch_size, freq_length(12), fit_length(30), index) / (batch_size, index)

    def __init__(self, TYPE = "all", index_set = [0, 1, 2, 3, 4]):
        """
            TYPE:
                all -> final global integrated model
                xgboost -> only xgboost way
                arima -> only arima way
                origin -> model on 5.9's pre
                delete_fc -> 取后60个TCN项
                lstm -> only DNN way
        """


        self.cpu = "cpu"
        self.gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.TYPE = TYPE

        self.Index_set = torch.tensor(index_set)
        self.Index_number = self.Index_set.shape[0]

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
        
        self.nn_model = NNModel(self.arima_model.onum, self.TYPE, self.Index_number) 
        self.best_nn_model = NNModel(self.arima_model.onum, self.TYPE, self.Index_number)  
        self.best_metric = 0   
        

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
        self.Index_set.to(device)

    def xgb_train(self, train_data, dev_data):
        train_data = train_data.transpose(0, 2, 1)
        dev_data = dev_data.transpose(0, 2, 1)
        for which_index in range(5):
            now_xgb = self.xgb[which_index]
            now_xgb.fit(train_data[:, :-1, which_index], train_data[:, -1, which_index], 
                eval_set = [(dev_data[:, :-1, which_index], dev_data[:, -1, which_index])], verbose=False)

    def xgb_test(self, test_data):
        right, tot = 0, 0
        test_data = test_data.transpose(0, 2, 1)
        for which_index in self.Index_set:
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
        for idx, which_index in enumerate(self.Index_set):
            now_xgb = self.xgb[which_index]
            now_data = infer_data[:, idx]
            # print(now_data.shape)
            predictions = now_xgb.predict(now_data)
            ret.append(predictions)
        ret = np.array(ret, dtype = np.float64).T
        # ret -> (batch_size, Index_number)
        ret = torch.tensor(ret).to(self.gpu)
        return ret

    def train(self, data, xgb_data, train_index, dev_index, device = "cpu", epochs = 100, learning_rate = 3 * 1e-3, loss_func = "MSE", weight_decay = 0.2, batch_size = 16, optimER = "sgd"):
        self.nn_model.to_device(device)
        self.best_nn_model.to_device(device)
        self.nn_model.train()
        self.epochs = epochs
        self.lr = learning_rate
        self.wd = weight_decay
        self.batch_size = batch_size
        self.optimizer_type = optimER
        train_data_raw, dev_data_raw = data[train_index], data[dev_index]

        self.xgb_train(xgb_data[train_index], xgb_data[dev_index])
        # self.xgb_test(xgb_data[dev_index])
        # exit(0)

        # train and update
        if loss_func == "MSE":
            criterion = nn.MSELoss().to(device) # Regression
        else:
            criterion = nn.BCELoss().to(device) # Classification

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

        self.train_epoch_loss, self.dev_epoch_loss = [], []
        for epoch in range(self.epochs):
            # transform the data -------> now combine the output of ARIMA Models
            train_data = self.ret_batch_data(train_data_raw, self.arima_model.get_ret_from_file(train_index))
            dev_data = self.ret_batch_data(dev_data_raw, self.arima_model.get_ret_from_file(dev_index))
            # move to device(GPU)
            self.preprocessing(train_data, device)
            self.preprocessing(dev_data, device)
            
            batch_loss, tot_sample = [], 0
            for batch_idx, (sel, x, arima_x, values) in enumerate(train_data):
                self.nn_model.zero_grad()
                Pred = self.nn_model(
                    x[:, :, self.Index_set], 
                    arima_x[:, :, self.Index_set], 
                    self.xgb_predict(
                        xgb_data[train_index][sel.cpu().numpy()][:, self.Index_set, :-1]
                    )
                ).reshape(-1)
                if loss_func == "MSE":
                    loss = criterion(Pred, values.reshape(x.shape[0], -1)[:, self.Index_set].reshape(-1)) * x.shape[0]
                else:
                    loss = criterion(nn.Sigmoid()(Pred), (values.reshape(x.shape[0], -1)[:, self.Index_set].reshape(-1) > 0).double()) * x.shape[0]
                try:
                    loss.backward()
                except:
                    global Print_XGboost
                    Print_XGboost += 1
                    if Print_XGboost == 1:
                        print("Now it's Xgboost, and do not need to Back Propagation")
                optimizer.step()
                batch_loss.append(loss.item() * self.Index_number)
                tot_sample += x.shape[0] * self.Index_number

            self.train_epoch_loss.append(sum(batch_loss) / tot_sample)
            print('(Training) Epoch: {} Average data point Loss(MSE): {:.6f}'.format(epoch, self.train_epoch_loss[-1]))

            # use the dev-data-set to test the model in independent dataset
            batch_loss, tot_sample = [], 0
            with torch.no_grad():
                tot, right = 0, 0
                for batch_idx, (sel, x, arima_x, values) in enumerate(dev_data):
                    pred = self.nn_model(
                        x[:, :, self.Index_set], 
                        arima_x[:, :, self.Index_set], 
                        self.xgb_predict(
                            xgb_data[dev_index][sel.cpu().numpy()][:, self.Index_set, :-1]
                        )
                    ).reshape(-1)
                    if loss_func == "MSE":
                        loss = criterion(pred, values.reshape(x.shape[0], -1)[:, self.Index_set].reshape(-1)) * x.shape[0]
                    else:
                        loss = criterion(nn.Sigmoid()(pred), (values.reshape(x.shape[0], -1)[:, self.Index_set].reshape(-1) > 0).double()) * x.shape[0]
                    
                    batch_loss.append(loss.item() * self.Index_number)
                    tot_sample += x.shape[0] * self.Index_number

                    tot += x.shape[0] * self.Index_number
                    right += torch.sum((pred * values.reshape(x.shape[0], -1)[:, self.Index_set].reshape(-1)) > 0)

                    # if batch_idx <= 2:
                    #     # print(x[:3, :, Index_set].reshape(3, -1))
                    #     print(pred[:3], values.reshape(x.shape[0], -1)[:, Index_set].reshape(-1)[:3], end = '\n')
                acc = right/tot
                # print('ACC in dev dataset: {:.6f}'.format(acc))

                self.dev_epoch_loss.append(sum(batch_loss) / tot_sample)
                print('(Dev Dataset) Epoch: {} Average data point Loss(MSE): {:.6f}'.format(epoch, self.dev_epoch_loss[-1]))

                # only save the best model !
                if acc >= self.best_metric :
                    self.best_metric = acc
                    self.best_nn_model.load_state_dict(self.nn_model.state_dict())
        
        print(f'total time = {time.time() - begin_time} s')

        # print(self.best_nn_model)

        return self.best_metric, self.train_epoch_loss, self.dev_epoch_loss

    def test(self, data, xgb_data, test_index, device = "cpu"):
        self.best_nn_model.to(device)
        self.best_nn_model.eval()

        # print(self.best_nn_model.state_dict())

        test_data = data[test_index]

        metrics = {
            # 'total_loss': 0.0,
            'TP': 0,
            'TN': 0,
            'FP': 0,
            'FN': 0,
            'acc': 0.0,
            'prec_1': 0.0,
            'recall_1': 0.0,
            'f1-score_1': 0.0,
            'prec_0': 0.0,
            'recall_0': 0.0,
            'f1-score_0': 0.0,
        }

        # criterion = nn.MSELoss().to(device) # Regression
        # criterion = nn.BCELoss().to(device) # Classification
        # transform the data
        test_data = self.ret_batch_data(test_data, self.arima_model.get_ret_from_file(test_index))
        # pre_move to GPU
        self.preprocessing(test_data, device)

        # NNN = None
        with torch.no_grad():
            for batch_idx, (sel, x, arima_x, values) in enumerate(test_data):
                # NNN = arima_x
                pred = self.best_nn_model(
                    x[:, :, self.Index_set], 
                    arima_x[:, :, self.Index_set], 
                    self.xgb_predict(
                        xgb_data[test_index][sel.cpu().numpy()][:, self.Index_set, :-1]
                    )
                ).reshape(-1)
                # print(pred.shape, values.reshape(x.shape[0], -1)[:, Index_set].reshape(-1).shape)
                # loss = criterion(pred, values.reshape(x.shape[0], -1)[:, self.Index_set].reshape(-1)) * x.shape[0]
                true_value = values.reshape(x.shape[0], -1)[:, self.Index_set].reshape(-1)

                # metrics['total_loss'] += loss.item() * self.Index_number
                metrics['TP'] += torch.sum((pred > 0) & (true_value > 0)).detach().cpu().numpy()
                metrics['TN'] += torch.sum((pred < 0) & (true_value < 0)).detach().cpu().numpy()
                metrics['FP'] += torch.sum((pred > 0) & (true_value < 0)).detach().cpu().numpy()
                metrics['FN'] += torch.sum((pred < 0) & (true_value > 0)).detach().cpu().numpy()


        metrics['acc'] = (metrics['TN'] + metrics['TP']) / (metrics['TN'] + metrics['TP'] + metrics['FN'] + metrics['FP'])
        metrics['prec_1'] = metrics['TP'] / (metrics['TP'] + metrics['FP'])
        metrics['recall_1'] = metrics['TP'] / (metrics['TP'] + metrics['FN'])
        metrics['f1-score_1'] = 2.0 / (1.0 / metrics['prec_1'] + 1.0 / metrics['recall_1'])
        metrics['prec_0'] = metrics['TN'] / (metrics['TN'] + metrics['FN'])
        metrics['recall_0'] = metrics['TN'] / (metrics['TN'] + metrics['FP'])
        metrics['f1-score_0'] = 2.0 / (1.0 / metrics['prec_0'] + 1.0 / metrics['recall_0'])
        return metrics