import os
import pandas as pd
import numpy as np
import covid_daily
from covid_daily.constants import AVAILABLE_CHARTS 


def records_country(country_name, data_folder):
    """
    Get and records data from Worldometers.info
    Args: 
        country_name: (str) name of the country to records data,
        data_folder: (str) folder path to records data in,
        add_dataset: (bool) if True, trigger renku dataset add command
    """
    data = [covid_daily.data(country=country_name, chart=chart, as_json=False) for chart in AVAILABLE_CHARTS]
    data = pd.concat(data, axis=1)
    csv_path = os.path.join(data_folder,"{0:s}.csv".format(country_name))
    data.to_csv(csv_path)
    return csv_path

def get_multivariate_sequence(dataframe, target_col, n_in, n_out):
    """
    convert a dataframe into "input" multivariate sequences and "output" univariate sequences.
    args:
        dataframe: (pandas DataFrame) data to use with DatetimeIndex
        target_col: (str) attribute name of the column to use as univariate "output" sequence
        n_in: (int) number of timesteps in "input" sequence
        n_out: (int) number of timesteps in "output" sequence
    returns: 
        X: (numpy array) array of "input" multivariate sequences
        Y: (numpy array) array of "input" univariate sequences
        in_last_date: (numpy array) array of index of the last timesteps from input sequences
    """
    df = dataframe.copy()
    df.dropna(inplace=True)
    data = df.values
    target_id = list(dataframe.columns).index(target_col)
    X, y, in_last_date = list(), list(), list()
    in_start = 0
    for current_index in range(len(data)):
        # define the end of the input sequence
        out_first = current_index + n_in
        out_last = out_first + n_out
        # ensure we have enough data for this instance
        if out_last <= len(data):
            x_input = data[current_index:out_first, :]
            X.append(x_input)
            y.append(data[out_first:out_last, target_id])
            in_last_date.append(df.index[out_first-1])
    return np.array(X), np.array(y), np.array(in_last_date)
