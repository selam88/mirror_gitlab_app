import os, sys
sys.path.append("../../")
sys.path.append("../")
sys.path.append("/app/mirror_gitlab_app/")
sys.path.append("/app/mirror_gitlab_app/src")
from src.utils.data import load_prediction_data, load_training_data
import pandas as pd
import numpy as np
import streamlit as st 
from datetime import timedelta
import altair as alt
from src.utils.train import *
from copy import deepcopy
from src.streamlit_app.dash_utils import *
from src.streamlit_app.chart_library import *


header = st.beta_container()
user_input = st.beta_container()
output_graphs = st.beta_container()
author_credits = st.beta_container()
# Hard-coded path, specific to streamlit app sharing
model_folder = "/app/mirror_gitlab_app/data/models/models_2/"
training_folder = "/app/mirror_gitlab_app/data/model-data/"
inference_folder = "/app/mirror_gitlab_app/data/inference-data/"
# Load model 
model = scaled_model(model_folder)

with header:
    st.title("Welcome to the Covid-19 New cases Predictions Application")
    st.markdown("""
    #### By: [Selim Amrari](https://www.linkedin.com/in/sélim-amrari08/)
    
    Welcome to the Covid-19 New cases Predictions tracker. This web application displays the daily number of new Covid-19 cases reported in the selected country as well as the prediction interval, since the beginning of the pandemic. This application also allows to display any predictions related to the selected date and the corresponding monthly average errors for each predicted timesteps.

    **Note:** If your browser display the dark theme by default, you can change it with to top right icon `>` Settings.
    
    **Note:** If you don't see the "User Selection" sidebar, please press the `>` icon on the top left side of your screen.
    
    """)

# Process predictions data
predictions, last_in_dates, country_array, overall_df = load_prediction_data(model_folder=training_folder, inference_folder=inference_folder)
input_seq, output_seq, _, _, _ = load_training_data(folder=training_folder, as_dataset=False, scale=False, shuffle=False)
datetime_index = [d for d in map(pd.to_datetime, last_in_dates)]
data_dic = {"infered_array": [v.ravel() for v in predictions], "country": country_array}
predictions_df = pd.DataFrame(data=data_dic, index=datetime_index)
country_list = list(predictions_df.country.unique())
displayed_country_list = [c.title() for c in country_list]

with user_input:
    st.sidebar.header('User Selection') 
    # Country widget
    country_ = st.sidebar.selectbox('Select Your Country:',displayed_country_list, index=4) 
    country = country_.lower()
    # Date widget
    #date = st.sidebar.date_input("Select the date you want to check", min_value=predictions_df.index[0], max_value=predictions_df.index[-1])
    date = st.sidebar.slider("Select the date you want to check", 
                             min_value=predictions_df.index[0].to_pydatetime(), 
                             max_value=predictions_df.index[-1].to_pydatetime(), 
                             value=pd.to_datetime("2020-09-07").to_pydatetime(), 
                             step=timedelta(days=1))
    noise = st.sidebar.slider("Select the amount of noise you want to add to the input sequence (in %)", 
                             min_value=0, 
                             max_value=100, 
                             value=10, 
                             step=2)
    date_timest = pd.to_datetime(date)
    
with output_graphs:
    # set up country df
    country_df = get_country_cases(overall_df, country)
    # set up prediction_df
    main_vis_df, country_pred, country_l_i_d = get_main_visualization_df(country, predictions, 
                                                                         last_in_dates, country_array, country_df)
    # set up start, end of month
    start, end = get_month_interval(date_timest)
    cutoff = pd.DataFrame({'start': [start], 'stop': [end]})
    # First chart : Create plot for entire period
    first_chart = get_first_chart(main_vis_df, country, cutoff, date_timest)
    first_chart
    # first and second chart note/explanation
    st.markdown("""
    **Note:** You can zoom on this graph if you are in front of a Desktop or Laptop by using your scrolling wheel on your mouse.
    
    The current model is predicting the "daily new covid cases" within the following 20 days. Two variables are used as intputs: the "daily new covid cases" and "daily amount of active covid cases" on the past 40 days. 
    
    The following chart displays the error predictions for the month corresponding to the selected date, averaged on each specific timesteps. It represents the expected error N-days after the last measurement """)
    # Prepare error chart
    country_outputs = output_seq[country_array==country]
    selected_pred, selected_outputs, selected_l_i_d = get_month_data([start, end], country_pred,
                                                      country_outputs, country_l_i_d)
    monthly_error_df = get_error_dataframe(selected_pred, selected_outputs)
    # Second chart : set error chart
    error_chart = get_second_chart(monthly_error_df, date_timest)
    error_chart
    
    # third and forth chart note/explanation
    st.markdown("""
    The following pair of charts display the situation specific to the date and the country selected. The left chart shows the true input and output sequence (Daily New Cases), and the obtained predictions. 
    
    The chart on the right side is aimed at evaluating the model stability. To this end, a specific amount of random noise is addded to the input sequence before triggering the predictions again. The chart allows to visualize the "Noisy" input sequence and to compare the "Noisy" predicted sequence with the original one.
    """)
    
    # Prepare third chart:  focus prediction chart
    selected_date_index = np.where(pd.to_datetime(country_l_i_d)==date_timest)[0][0]
    country_input = input_seq[country_array==country]
    focused_df = get_focused_predictions_df(selected_date_index, country_l_i_d, country_input, country_outputs, country_pred)
    noisy_focus_df = get_noisy_focused_predictions_df(focused_df, noise, country_input[selected_date_index], model)
    multi_seg_df = pd.DataFrame(focused_df.stack()).reset_index()
    noisy_seg_df = pd.DataFrame(noisy_focus_df.stack()).reset_index()
    renaming = {"level_0":"Date", "level_1":"Category", 0:"Daily new cases"}
    multi_seg_df.rename(columns=renaming, inplace=True)
    noisy_seg_df.rename(columns=renaming, inplace=True)
    third_chart, forth_chart = get_third_chart(multi_seg_df, date_timest), get_third_chart(noisy_seg_df, date_timest, noisy=True)
    third_chart | forth_chart
    st.markdown("""
    **Note:** You can change the amount of noise added to the input sequence, using the "User Selection" sidebar.
    """)
    
with author_credits:
    st.header(f'Credits')
    st.markdown("""
    **Thank you for using my application!**
    
    This dashboard has been developed thanks to the following [Covid-19-County-Tracker app](https://github.com/cerratom/Covid-19-County-Tracker) used as template. Feel free to have a look on the original [dashboard app](https://share.streamlit.io/cerratom/covid-19-county-tracker/county.py)
    
    The dataset used to feed this application is provided by [Wordlometers](https://www.worldometers.info/coronavirus/) and ingested thanks to the [covid-daily python API](https://pypi.org/project/covid-daily/).

    This application uses the [Streamlit package library](https://streamlit.io). Please find some more examples of Streamlit apps in the [official Streamlit gallery](https://streamlit.io/gallery) 
    """)


