import os
import pandas as pd
import numpy as np
import covid_daily
from covid_daily.constants import AVAILABLE_CHARTS 
import pickle


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

def get_multivariate_sequence(dataframe, target_col, n_in, n_out, convert_date=True):
    """
    convert a dataframe into "input" multivariate sequences and "output" univariate sequences.
    args:
        dataframe: (pandas DataFrame) data to use with DatetimeIndex
        target_col: (str) attribute name of the column to use as univariate "output" sequence
        n_in: (int) number of timesteps in "input" sequence
        n_out: (int) number of timesteps in "output" sequence
        convert_date: (bool) if True, convert DatetimeIndex into string
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
    if convert_date:
        in_last_date = [d.strftime("%Y-%m-%d") for d in in_last_date]
    return np.array(X), np.array(y), np.array(in_last_date)

def save_obj(obj, name):
    '''
    save object
    '''
    name = name+".pkl" if not name.endswith(".pkl") else name
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(name):
    '''
    load object
    '''
    import pdb; pdb.set_trace()
    name = name if not name.endswith(".pkl") else name
    with open(name, 'rb') as f:
        return pickle.load(f)
    
def load_training_data(model_folder="/work/test-first-project/data/model-data/"):
    """
    load training data from hard-coded directory
    args:
        model_folder(optional): (str)path to the folder containing saved data
    return: 
        input_seq: (numpy array) array of "input" multivariate sequences
        output_seq: (numpy array) array of "input" univariate sequences
        last_in_dates: (numpy array) array of index of the last timesteps from input sequences
        country_array: (numpy array) array of countries corresponding to the sequences 
    """
    input_seq = load_obj(os.path.join(model_folder, "input_sequences.pkl"))
    output_seq = load_obj(os.path.join(model_folder, "output_sequences.pkl"))
    last_in_dates = load_obj(os.path.join(model_folder, "last_in_dates.pkl"))
    country_array = load_obj(os.path.join(model_folder, "country.pkl"))
    return input_seq, output_seq, last_in_dates, country_array