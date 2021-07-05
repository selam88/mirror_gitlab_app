import os, sys
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from sklearn.preprocessing import StandardScaler
from utils.data import save_obj, load_obj


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

def scale_data(X_seq, Y_seq=None, scaler=None, target_id=1):
    """
    scale sequence data
    args:
        X_seq: (numpy array) input sequences
        Y_seq: (numpy array) output sequences
        scaler: (sklearn.preprocessing.StandardScaler) fitted StandardScaler instance
        target_id: (int) index of the target variable
    return:
        X_seq: (numpy array) scaled input sequences
        Y_seq: (numpy array) scaled output sequences
        scaler: (sklearn.preprocessing.StandardScaler) StandardScaler instance
    """
    is isinstance(scaler, type(None)):
        scaler = StandardScaler()
        scaler.fit(X_seq[:,-1,:])
    for f in range(X_seq.shape[2]):
        X_seq[:,:,f] = (X_seq[:,:,f] - scaler.mean_[f]) / scaler.scale_[f]
    if not isinstance(Y_seq, type(None)):
        Y_seq = (Y_seq - scaler.mean_[target_id]) / scaler.scale_[target_id]
        return X_seq, Y_seq, scaler
    return X_seq, scaler

def unscale_data(scaler, X_seq=None, Y_seq=None, target_id=1):
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
    if not isinstance(X_seq, type(None)):
        for f in range(X_seq.shape[2]):
            X_seq[:,:,f] = (X_seq[:,:,f] * scaler.scale_[f]) + scaler.mean_[f]
        if isinstance(Y_seq, type(None)):
            return X_seq
    if not isinstance(Y_seq, type(None)):
        Y_seq = (Y_seq * scaler.scale_[target_id]) + scaler.mean_[target_id]
        if isinstance(X_seq, type(None)):
            return Y_seq
    return X_seq, Y_seq
    
class scaled_model:
    """
    Sequence model used with scaling as preprocessing
    """
    def __init__(self, model_folder):
        """
        args:
            model_folder: (str) folder containing saved model
        """
        self.score_path = os.path.join(model_folder, "score.csv")
        self.model_path = os.path.join(model_folder, "model")
        self.scaler_path = os.path.join(model_folder, "scaler.pkl")
        self.load()
        return
        
    def load(self):
        """
        load data from the model
        """
        self.model = load_model(self.model_path)
        self.scaler = load_obj(self.scaler_path)
        self.history = pd.read_csv(self.score_path)
        
    def predict(X_seq, preprocess=True):
        """
        Apply inference
        args:
            X_seq: (numpy array) inputs sequences to infer
            preprocessing: (bool) if True, X_seq are scale before inference
        return:
            predictions: (numpy array) infered sequences after unscaling
        """
        if preprocess:
            X_seq, _ = scale_data(X_seq, scaler=self.scaler)
        standard_predictions = self.model.predict(X_seq)
        predictions = unscale_data(self.scaler, Y_seq=standard_predictions)
        return predictions
        