import os
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed


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

def save_model_score(folder_path, model):
    """
    save model in a dedicated folder. Records fitting history as csv.
    if already exists, only append new history results and overwrite model.
    args:
        folder_path: (str) path to the dedicated folder to record model in
        model: (tf.Sequential) model instance to records
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
    return