# Welcome to the Covid-19 New cases Predictions Application

**FEEL FREE TO DICOSVER THE [APP](https://share.streamlit.io/selam88/mirror_gitlab_app/src/streamlit_app/covid_performance_tracker.py)**

## Introduction

The purpose of this project is to build a dashboarding application 
allowing to browse COVID Daily New Cases predictions back in time. 
The predictions are based on a multi-timesteps prediction model, taking
as feature variables the COVID Daily Active Cases and the COVID Daily
New Cases from the past 40 days. COVID Daily New Cases are predicted 
up to 20 days next.

## Working with the project

The simplest way to start your project is right from the Renku
platform - just click on the `Environments` tab and start a new session.
This will start an interactive environment right in your browser.

To work with the project anywhere outside the Renku platform,
click the `Settings` tab where you will find the
git repo URLs - use `git` to clone the project on whichever machine you want.

### Changing interactive environment dependencies

Initially we install a very minimal set of packages to keep the images small.
However, you can add python and conda packages in `requirements.txt` and
`environment.yml` to your heart's content. If you need more fine-grained
control over your environment, please see [the documentation](https://renku.readthedocs.io/en/latest/user/advanced_interfaces.html#dockerfile-modifications).

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
