import os, sys
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from sklearn.preprocessing import StandardScaler
from utils.data import save_obj


def set_MVar_EncDec_lstm(in_timesteps, out_timesteps, n_features, n_units=200):
    """
    Set up a multivariate auto-encoder model for multi-timesteps forcasting.
    args:
        in_timesteps: (int) number of timesteps in multivariate input sequences
        out_timesteps: (int) number of timesteps in univariate output sequences
        n_variables: (int) number of features in input sequences
        n_units: (int) number of units in the LSTM layer
    return:
        model: (tf.Sequential) model instance
    """
    
    # define model
    model = Sequential()
    model.add(LSTM(n_units, activation='relu', input_shape=(in_timesteps, n_features)))
    model.add(RepeatVector(out_timesteps))
    model.add(LSTM(n_units, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(int(n_units/2), activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adam')
    return model

def save_model_score(folder_path, model, scaler=None):
    """
    save model in a dedicated folder. Records fitting history as csv.
    if already exists, only append new history results and overwrite model.
    args:
        folder_path: (str) path to the dedicated folder to record model in
        model: (tf.Sequential) model instance to records
        scaler(optional): scaler used to train
    """
    if not os.path.exists(folder_path): 
        os.mkdir(folder_path)
    model_path = os.path.join(folder_path, "model")
    csv_path = os.path.join(folder_path, "score.csv")
    model.save(model_path, overwrite=True, include_optimizer=True)
    df = pd.DataFrame(model.history.history)
    if os.path.exists(csv_path):
        print("model already exists: append history")
        old_df = pd.read_csv(csv_path)
        df = pd.concat([old_df, df], axis=0)
    df.to_csv(csv_path, index=False)  
    if not isinstance(scaler, type(None)):
        save_obj(scaler, os.path.join(folder_path, "scaler.pkl"))
    return

def scale_data(X_seq, Y_seq, target_id=1):
    """
    scale sequence data
    args:
        X_seq: (numpy array) input sequences
        Y_seq: (numpy array) output sequences
        target_id: (int) index of the target variable
    return:
        X_seq: (numpy array) scaled input sequences
        Y_seq: (numpy array) scaled output sequences
        scaler: (sklearn.preprocessing.StandardScaler) StandardScaler instance
    """
    scaler = StandardScaler()
    scaler.fit(X_seq[:,-1,:])
    for f in range(X_seq.shape[2]):
        X_seq[:,:,f] = (X_seq[:,:,f] - scaler.mean_[f]) / scaler.scale_[f]
    Y_seq = (Y_seq - scaler.mean_[target_id]) / scaler.scale_[target_id]
    return X_seq, Y_seq, scaler

def unscale_data(X_seq, Y_seq, scaler, target_id=1):
    """
    reverse scaling of sequence data
    args:
        X_seq: (numpy array) scaled input sequences
        Y_seq: (numpy array) scaled output sequences
        scaler: (sklearn.preprocessing.StandardScaler) StandardScaler instance
        target_id: (int) index of the target variable
    return:
        X_seq: (numpy array) input sequences
        Y_seq: (numpy array) output sequences
    """
    for f in range(X_seq.shape[2]):
        X_seq[:,:,f] = (X_seq[:,:,f] * scaler.scale_[f]) + scaler.mean_[f]
    Y_seq = (Y_seq * scaler.scale_[target_id]) + scaler.mean_[target_id]
    return X_seq, Y_seq
    