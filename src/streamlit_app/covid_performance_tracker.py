import os, sys
sys.path.append("/work/test-first-project/src")
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from utils.data import load_prediction_data, load_training_data
import streamlit as st 
from datetime import timedelta
import altair as alt

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

def get_main_visualization_df(country, predictions, last_in_dates, country_array):
    """
    create visualization dataframe containing measurement as well as maximum and minimum
    predictions for each date.
    args:
        country: (str) selected country
        predictions: (numpy array) array of predicted sequences
        last_in_dates: (numpy array) date array corresponding to the last "input" timesteps
        country_array: (numpy array) country name array of the predicted sequences
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

header = st.beta_container()
user_input = st.beta_container()
output_graphs = st.beta_container()
author_credits = st.beta_container()

with header:
    st.title("Welcome to the Covid-19 New cases Predictions Application")
    st.markdown("""
    #### By: [Selim Amrari]( www.linkedin.com/in/sélim-amrari08/)
    
    Welcome to the Covid-19 New cases Predictions tracker. This web application displays the daily number of new Covid-19 cases reported in the selected country as well as the prediction interval, since the beginning of the pandemic. This application also allows to display any predictions related to the selected date and the corresponding monthly average errors for each predicted timesteps.

    **Note:** If you don't see the "User Selection" sidebar, please press the `>` icon on the top left side of your screen.
    
    """)

# Process predictions data
predictions, last_in_dates, country_array, overall_df = load_prediction_data()
input_seq, output_seq, _, _ = load_training_data()
datetime_index = [d for d in map(pd.to_datetime, last_in_dates)]
data_dic = {"infered_array": [v.ravel() for v in predictions], "country": country_array}
predictions_df = pd.DataFrame(data=data_dic, index=datetime_index)
country_list = list(predictions_df.country.unique())

with user_input:
    st.sidebar.header('User Selection') 
    
    # Country widget
    country = st.sidebar.selectbox('Select Your Country:',country_list) 
    
    # Date widget
    #date = st.sidebar.date_input("Select the date you want to check", min_value=predictions_df.index[0], max_value=predictions_df.index[-1])
    date = st.sidebar.slider("Select the date you want to check", 
                             min_value=predictions_df.index[0].to_pydatetime(), 
                             max_value=predictions_df.index[-1].to_pydatetime(), 
                             value=predictions_df.index[0].to_pydatetime(), 
                             step=timedelta(days=1))
    date_timest = pd.to_datetime(date)
    
    
with output_graphs:

    # set up country df
    country_df = get_country_cases(overall_df, country)
    # set up prediction_df
    main_vis_df, country_pred, country_l_i_d = get_main_visualization_df(country, predictions, last_in_dates, country_array)
    # set up start, end of month
    start, end = get_month_interval(date_timest)
    cutoff = pd.DataFrame({'start': [start], 'stop': [end]})
    # Create plot for entire period
    # shaded area
    area = alt.Chart(main_vis_df).mark_area(opacity=0.5, color='#cb181d').encode(
        x=alt.X('index', axis=alt.Axis(title='Date')),
        y=alt.Y('Min'), y2=alt.Y2('Max')
    ).properties(width=800, height=350).interactive()
    # main curve
    line = alt.Chart(main_vis_df).mark_line(color="#0868ac").encode(
        x=alt.X('index', axis=alt.Axis(title='Date')),
        y='Daily new cases'
    ).properties(width=800, height=350).interactive()
    # shaded box
    month_box = alt.Chart(
        cutoff.reset_index()
    ).mark_rect(
        opacity=0.2
    ).encode(
        x='start',
        x2='stop',
        y=alt.value(0),  # pixels from top
        y2=alt.value(350),  # pixels from top
        color='index:N'
    ).properties(width=800, height=350).interactive()
    # display
    area+line+month_box
    
    # set error chart
    country_outputs = output_seq[country_array==country]
    selected_pred, selected_outputs, selected_l_i_d = get_month_data([start, end], country_pred,
                                                      country_outputs, country_l_i_d)
    monthly_error_df = get_error_dataframe(selected_pred, selected_outputs)
    error_chart = alt.Chart(monthly_error_df).transform_density(
        'error',
        as_=['error', 'density'],
        extent=[monthly_error_df.error.min() -100, monthly_error_df.error.max()+100],
        groupby=['Daily timestep']
    ).mark_area(orient='horizontal').encode(
        y='error:Q',
        color=alt.Color(
            'Daily timestep:N',
            scale=alt.Scale(scheme='plasma')),
        x=alt.X(
            'density:Q',
            stack='center',
            impute=None,
            title=None,
            axis=alt.Axis(labels=False, values=[0],grid=False, ticks=True),
        ),
        column=alt.Column(
            'Daily timestep:N',
            header=alt.Header(
                titleOrient='bottom',
                labelOrient='bottom',
                labelPadding=0,
            ),
        )
    ).properties(
        width=35
    ).configure_facet(
        spacing=0
    ).configure_view(
        stroke=None
    )
    error_chart
    
    st.markdown("""**Note:** You can zoom on this graph if you are in front of a Desktop or Laptop by using your scrolling wheel on your mouse.""")

with author_credits:
    st.header(f'Credits')
    st.markdown("""
    **Thank you for using my application!**
    
    This dashboard has been devlopped thanks to the following [Covid-19-County-Tracker app](https://github.com/cerratom/Covid-19-County-Tracker) used as template.
    
    The dataset used to feed this application is provided by [Wordlometers](https://www.worldometers.info/coronavirus/) and ingested thanks to the [covid-daily python API](https://pypi.org/project/covid-daily/).

    This application uses the [Streamlit package library](https://streamlit.io). Pleas find some more examples of Streamlit apps in the [official Streamlit gallery](https://streamlit.io/gallery) 
    """)


