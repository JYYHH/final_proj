import numpy as np
import pandas as pd


def sum_up(data, interval):
    return np.array([np.sum(data[i:i+interval], axis = 0) for i in range(0, data.shape[0], interval)])

def trans(raw_data, length = 931, interval = 1, after = 30): # 931
    # raw_data (total_length, Index_number)
    # It's best that `length` - `after` is multiplier of `interval`
    mod = (length - after) % interval
    Dim0 = raw_data.shape[0] - length + 1
    return np.array(
            [
                np.vstack((
                    sum_up(raw_data[i + mod : i + length - after], interval), 
                    np.sum(raw_data[i + length - after : i + length], axis = 0, keepdims = True)
                )) 
                    for i in range(Dim0)
            ]
        ).transpose(0, 2, 1) # `interval` days interval and `after` days prediction
    # shape -> (data_length, Index_number, new_length_with_last_postion_label)

