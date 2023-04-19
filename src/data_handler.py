import numpy as np
import pandas as pd

def trans(raw_data, length = 930 + 1):
    """ Rebuild our data into next word prediction shape """
    Dim0 = raw_data.shape[0] - length + 1
    return np.array([raw_data[i : i + length] for i in range(Dim0)]).transpose(0, 2, 1)
    # shape -> (data_length, index, length)

