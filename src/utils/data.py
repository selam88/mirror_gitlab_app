import os, json, pickle, covid_daily
import pandas as pd
import numpy as np
from covid_daily.constants import AVAILABLE_CHARTS 
from datetime import date
import tensorflow as tf
from scipy.ndimage import maximum_filter
from copy import deepcopy

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
    name = name+".pkl" if not name.endswith(".pkl") else name
    with open(name, 'rb') as f:
        return pickle.load(f)
    
def augment_data(train_seq, target_set, percent_noise=5.0):
    """add random noise to input sequences"""
    new_train_seq = deepcopy(train_seq.numpy())
    max_noise = maximum_filter(new_train_seq, size=(15,1), mode="nearest")
    max_noise = np.round(max_noise*(percent_noise/100.0)).astype(np.int)
    max_noise[max_noise<1] = 1
    assert np.sum(max_noise<1)==0
    noise = np.random.randint(-1*max_noise, max_noise) 
    new_train_seq = new_train_seq + noise
    new_train_seq[new_train_seq<0] = 0
    return tf.convert_to_tensor(new_train_seq), target_set

@tf.function
def set_shape(x,y, shape_x, shape_y):
    """set tensors shape"""
    x = tf.ensure_shape(x, (40,2))
    y = tf.ensure_shape(y, (20,1))
    return x, y
    
def load_training_data(folder="/work/test-first-project/data/model-data", 
                       as_dataset=False, scale=True, do_augment=True,
                      shuffle=True, validation_split=0.08, batch_size=512):
    """
    load training data from hard-coded directory
    args:
        folder(optional): (str)path to the folder containing saved data
    return: 
        input_seq: (numpy array) array of "input" multivariate sequences
        output_seq: (numpy array) array of "input" univariate sequences
        last_in_dates: (numpy array) array of index of the last timesteps from input sequences
        country_array: (numpy array) array of countries corresponding to the sequences 
    """
    from .train import scale_data
    
    input_seq = load_obj(os.path.join(folder, "input_sequences.pkl"))
    output_seq = load_obj(os.path.join(folder, "output_sequences.pkl"))
    last_in_dates = load_obj(os.path.join(folder, "last_in_dates.pkl"))
    country_array = load_obj(os.path.join(folder, "country.pkl"))
    if shuffle:
        np.random.seed(42)
        permute_id = np.random.permutation(input_seq.shape[0])
        input_seq, output_seq = input_seq[permute_id], output_seq[permute_id]
        last_in_dates, country_array = last_in_dates[permute_id], country_array[permute_id]
    if scale:
        input_seq, output_seq, scaler = scale_data(input_seq, output_seq)
    else:
        scaler = None
    if as_dataset:
        dataset = tf.data.Dataset.from_tensor_slices((input_seq, output_seq))
        if do_augment:
            dataset = dataset.map(lambda x, y: tf.py_function(augment_data, [x,y], [tf.float64, tf.float64]),
                                 num_parallel_calls=tf.data.experimental.AUTOTUNE)
        shape_fn = lambda x,y: set_shape(x, y, input_seq.shape[1:], output_seq.shape[1:])
        dataset = dataset.map(shape_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # splitting validation data
        #dataset = dataset.shuffle(len(dataset), seed=8)
        train_size = int((1-validation_split) * len(dataset))
        train_ds = dataset.take(train_size)    
        val_ds = dataset.skip(train_size)
        train_ds = train_ds.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        val_ds = val_ds.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return train_ds, val_ds, last_in_dates, country_array, scaler
    else:
        return input_seq, output_seq, last_in_dates, country_array, scaler

def load_prediction_data(model_folder="/work/test-first-project/data/model-data/", 
                         inference_folder="/work/test-first-project/data/inference-data"):
    """
    load predictions data from hard-coded directory
    args:
        model_folder(optional): (str)path to the folder containing saved data
    return: 
        predictions: (numpy array) array of predicted sequences
        last_in_dates: (numpy array) array of index of the last timesteps from input sequences
        country_array: (numpy array) array of countries corresponding to the sequences 
        overall_df: (pandas DataFrame) overall dataframe of new cases 
    """
    predictions = load_obj(os.path.join(inference_folder, "predictions.pkl"))
    last_in_dates = load_obj(os.path.join(inference_folder, "last_in_dates.pkl"))
    country_array = load_obj(os.path.join(inference_folder, "country.pkl"))
    overall_df = pd.read_csv(os.path.join(model_folder, "overall_cases.csv"), index_col="Date", parse_dates=True)
    return predictions, last_in_dates, country_array, overall_df

def records_details(location_path, details_dic, append_current_date=True):
    """
    Records a json file reporting details of the saved data
    args:
        location_path: (str) path of the json to record
        details_dic: (dic) dictionnary of the details to record
        append_current_date: (bool) if True, add currend date as processing date
    return: 
        None
    """
    if not location_path.endswith(".json"):
        location_path = os.path.join(location_path, "details.json")
    if append_current_date:
        details_dic["processing_date"] = [date.today().strftime("%Y-%m-%d")]
    with open(location_path, 'w') as output:
        json.dump(details_dic, output, indent=4)
    return

def reads_details(location_path):
    """
    Read a json file reporting details of the saved data
    args:
        location_path: (str) path of the json to parse
    return: 
        details_dic: (dic) dictionnary of the details to record
    """
    if not location_path.endswith(".json"):
        location_path = os.path.join(location_path, "details.json")
    with open(location_path, 'r') as output:
        details_dic = json.load(output)
    return details_dic
    