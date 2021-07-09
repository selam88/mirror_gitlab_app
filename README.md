# Welcome to the Covid-19 New cases Predictions Application

![Alt text](https://renkulab.io/gitlab/selim.amrari.pro/test-first-project/-/raw/master/doc/stream_app_large.gif)

**FEEL FREE TO DICOSVER THE [APP](https://share.streamlit.io/selam88/mirror_gitlab_app/src/streamlit_app/covid_performance_tracker.py)**

## Introduction

The purpose of this project is to build a dashboarding application 
allowing to browse COVID Daily New Cases predictions back in time. 
The predictions are based on a multi-timesteps prediction model, taking
as feature variables the COVID Daily Active Cases and the COVID Daily
New Cases from the past 40 days. COVID Daily New Cases are predicted 
up to 20 days next.

This project is meant to be automaticaly updated on a regulat basis, 
retrieving daily COVID data from the [Wordlometers](https://www.worldometers.info/coronavirus/) website
through the use the [covid-daily python API](https://pypi.org/project/covid-daily/).

## Working with the project

The implementation of the project relys on 4 scripts which allow
to aumatize specific parts of the project. This 4 application
script, located in src/ are:
 
	* download_data.py: download and store all available data from [Wordlometers](https://www.worldometers.info/coronavirus/)

	* format_data.py: preprocess downloaded COVID data into trainable multivariates sequences

	* train.py: train a model composed of LongShortTermMemory layers with parameters-adjustable architecture

	* infer.py: apply inference and store the predictions to be displayed on the web app

Lastly, the dashboarding application is defined within a dedicated app folder: src/streamlit_app

### Working with the peoject - Renku command automatization

You can update the data which are displayed on the dashboard by running the following command: 

	* bash src/bash_sc/update_data.sh
   
it will automatically download, format, and infer new available data.

You can retrain the model with new available data by running the following command: 

	* bash src/bash_sc/update_and_retrain.sh
   
it will automatically download, format, re-train and infer new available data.

## Project configuration

Project options can be found in `.renku/renku.ini`. In this
project there is currently only one option, which specifies
the default type of environment to open, in this case `/lab` for
JupyterLab. You may also choose `/tree` to get to the "classic" Jupyter
interface.

## credits

**Thank you for using my application!**
    
This dashboard has been devlopped thanks to the following [Covid-19-County-Tracker app](https://github.com/cerratom/Covid-19-County-Tracker) used as template. Feel free to have a look on the original [dashboard app](https://share.streamlit.io/cerratom/covid-19-county-tracker/county.py)
    
The dataset used to feed this application is provided by [Wordlometers](https://www.worldometers.info/coronavirus/) and ingested thanks to the [covid-daily python API](https://pypi.org/project/covid-daily/).

This application uses the [Streamlit package library](https://streamlit.io). Please find some more examples of Streamlit apps in the [official Streamlit gallery](https://streamlit.io/gallery) 
