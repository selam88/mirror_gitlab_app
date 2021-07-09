import os, sys
import pandas as pd
import numpy as np
import streamlit as st 
from datetime import timedelta
import altair as alt
from copy import deepcopy


def get_country_cases(overall_df, country):
    """
    create dataframe from the contry to visualize
    args:
        overall_df: (pandas dataFrame) dataframe containing cases of all available countries
        country: (str) name of the selected country
    return:
        country_df: (pandas dataFrame) dataframe containing cases of the selected country
    """
    country_df = overall_df[[c for c in overall_df.columns if c.startswith(country)]].copy()
    country_df.rename(columns={country_df.columns[0]:"Daily new cases"}, inplace=True)
    country_df.dropna(inplace=True)
    return country_df

def get_main_visualization_df(country, predictions, last_in_dates, country_array, country_df):
    """
    create visualization dataframe containing measurement as well as maximum and minimum
    predictions for each date.
    args:
        country: (str) selected country
        predictions: (numpy array) array of predicted sequences
        last_in_dates: (numpy array) date array corresponding to the last "input" timesteps
        country_array: (numpy array) country name array of the predicted sequences
        country_df: (pandas dataFrame) dataframe containing ALL cases from the selected country
    returns:
        visualization_df: (pandas dataframe) dataframe with data to display
        country_predictions: (numpy array) predicted sequences array of the selected country
        country_last_in_dates: (numpy array) date array from the selected country 
                                corresponding to the last "input" timesteps
    """
     # set up overall prediction_df for country
    predictions_dic = {"date":[], "value":[]}
    country_predictions = predictions[country_array==country]
    country_last_in_dates = last_in_dates[country_array==country]
    country_last_in_datetime = pd.to_datetime(country_last_in_dates)
    for i in range(country_predictions.shape[1]):   
        new_points = country_predictions[:, i, 0]
        new_dates = country_last_in_datetime + timedelta(days=i+1)
        predictions_dic["date"].extend(new_dates)
        predictions_dic["value"].extend(new_points)
    country_prediction_df = pd.DataFrame(data=predictions_dic)
    country_prediction_df.reset_index(inplace=True)
    interval_df_min = country_prediction_df.groupby("date").min().rename(columns={"value":"Min"})
    interval_df_max = country_prediction_df.groupby("date").max().rename(columns={"value":"Max"})
    assert all(interval_df_min.index==interval_df_max.index)
    interval_df = pd.concat([interval_df_min["Min"], interval_df_max["Max"]], axis=1)
    # merge measurement and predicitons
    visualization_df = pd.concat([country_df, interval_df], axis=1)
    visualization_df.reset_index(inplace=True)
    return visualization_df, country_predictions, country_last_in_dates

def get_month_interval(date_timestamp):
    """
    return the first and last day of the month corresponding to the given timestamp
    args:
        date_timestamp: (pandas Timestamp) timestamp within the month to define
    return:
        start: (pandas Timestamp) timestamp of the first day of the month
        end: (pandas Timestamp) timestamp of the last day of the month
    """
    start = date_timestamp - timedelta(days=date_timestamp.day-1)
    end = date_timestamp + timedelta(days=date_timestamp.daysinmonth-date_timestamp.day)
    return start, end

def get_month_data(interval, country_pred, country_outputs, country_l_i_d):

    """
    return the predictions data corresponding to the selected interval
    args:
        interval: (Timestamp tuple) tuple of timestamp with period start and end
        country_pred: (numpy array) predicted sequences array of the selected country
        country_outputs: (numpy array) outputs sequences array of the selected country
        country_l_i_d: (numpy array) date array from the selected country corresponding
                        to the last "input" timesteps
    return:
        selected_pred: (numpy array) selected predicted sequences array
        selected_l_i_d:  (numpy array) date array from the selected period
        selected_outputs: (numpy array) selected outputs sequences array
    """
    start, end = interval
    country_l_i_tmstp =  pd.to_datetime(country_l_i_d)
    interval_ind = np.logical_and(country_l_i_tmstp>start, country_l_i_tmstp<end)
    selected_pred = country_pred[interval_ind]
    selected_l_i_d = country_l_i_d[interval_ind]
    selected_outputs = country_outputs[interval_ind]
    return selected_pred, selected_outputs, selected_l_i_d

def get_error_dataframe(predictions, output_seq):
    """
    compute a monthly prediction error per timesteps
    args:
        predictions: (numpy array) array of predicted sequences
        output_seq: (numpy array) array of outputs sequences
    returns:
        error_df: (pandas dataframe) dataframe of error
    """
    data_dic = {"Daily timestep": [], "error": [], "order":[]}
    for i in range(predictions.shape[1]):
        data_dic["error"].extend((predictions[:, i, :] - output_seq[:, i, :]).ravel())
        data_dic["Daily timestep"].extend(np.full((predictions.shape[0],), i+1))
        data_dic["order"].extend(np.full((predictions.shape[0],), i+1))
    error_df = pd.DataFrame(data=data_dic)
    return error_df

def get_focused_predictions_df(i, last_in_dates, input_seq, output_seq, predictions):
    """
    Build the DataFrame contaning input, output, predicted sequences to plot
    args:
        i: (int) index of the date to plot
        input_seq: (numpy array) array of input sequences
        output_seq: (numpy array) array of input sequences
        last_in_dates: (numpy array) date array corresponding to the last "input" timesteps
        predictions: (numpy array) array of predicted sequences
    returns:
        data_df: (pandas DataFrame) contaning input, output, predicted sequences to plot
    """
    current_date = pd.to_datetime(last_in_dates[i])
    in_dim, out_dim = input_seq.shape[1], output_seq.shape[1]
    dates = pd.date_range(start=current_date-timedelta(days=in_dim-1),
                          end=current_date+timedelta(days=out_dim))
    data_dic = {}
    data_dic["inputs sequence"] = np.concatenate([input_seq[i, :, 1], np.full((out_dim,), np.nan)])
    data_dic["true value"] = np.concatenate([np.full((in_dim-1,), np.nan), np.array([input_seq[i, -1, 1]]), output_seq[i, :, 0]])
    data_dic["predictions"] = np.concatenate([np.full((in_dim-1,), np.nan), np.array([input_seq[i, -1, 1]]), predictions[i, :, 0]])
    data_df = pd.DataFrame(data=data_dic, index=dates)
    return data_df

def get_noisy_focused_predictions_df(multi_seg_df_, percent_noise, in_seq, model):
    """
    Build the DataFrame contaning input, output, predicted sequences as well as
    the noisy input and predicted sequences to plot.
    args:
        data_df: (pandas DataFrame) contaning input, output, predicted sequences to plot
        percent_noise: (int) percentage of noise to add to the input sequences
        in_seq: (numpy array) array of the UNIQUE input sequence to infer
        model: (scaled model) model instance to use for inference
        predi"""
    multi_seg_df = multi_seg_df_.copy()
    in_dim, out_dim = in_seq.shape[0], (~multi_seg_df["predictions"].isna()).sum()-1
    max_noise = [round((percent_noise/100.0)* np.max(in_seq[:, i])) for i in [0, 1]]
    new_input_seq = deepcopy(in_seq)
    noise = [np.random.randint(-1*max_noise[i], max_noise[i], in_seq.shape[0]) for i in [0, 1]]
    for i in [0, 1]:
        new_input_seq[:,i] = new_input_seq[:,i] + noise[i]
        new_input_seq[new_input_seq[:,i]<0,i] = 0
    new_predictions = model.predict(new_input_seq[np.newaxis, :, :])
    multi_seg_df["noisy_predictions"] = np.concatenate([np.full((in_dim-1,), np.nan), np.array([new_input_seq[-1, 1]]), new_predictions[0, :, 0]])
    multi_seg_df["noisy inputs sequence"] = np.concatenate([new_input_seq[:, 1], np.full((out_dim,), np.nan)])
    return multi_seg_df
