import numpy as np
import pandas as pd


def sum_up(data, interval):
    return np.array([np.sum(data[i:i+interval], axis = 0) for i in range(0, data.shape[0] - interval + 1, interval)])

def trans(raw_data, length = 931, interval = 1): # 931
    """ Rebuild our data into next word prediction shape """
    # raw_data (total_length, Index_number)

    # some data transformation
    # raw_data = raw_data / np.fabs(raw_data).max(axis = 0, keepdims = True)
    # mid_data (total_length, Index_number)

    Dim0 = raw_data.shape[0] - length + 1
    # return np.array([np.vstack((raw_data[i : i + length - interval], np.sum(raw_data[i + length - interval : i + length], axis = 0, keepdims = True))) for i in range(Dim0)]).transpose(0, 2, 1) # 1 days interval and `interval` days prediction
    return np.array([ sum_up(raw_data[i : i + length], interval) for i in range(Dim0)]).transpose(0, 2, 1)
    # shape -> (data_length, index, length)

