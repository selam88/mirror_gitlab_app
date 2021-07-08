# Welcome to the Covid-19 New cases Predictions Application

FEEL FREE TO DICOSVER THE [APP](https://share.streamlit.io/selam88/mirror_gitlab_app/src/streamlit_app/covid_performance_tracker.py)

## Introduction

The purpose of this project is to build a dashboarding application 
allowing to browse COVID Daily New Cases predictions back in time. 
The predictions are based on multi-timesteps predictions model, taking
as feature variables the COVID Daily Active Cases and the COVID Daily
New Cases from the past 40 days. COVID Daily New Cases are predicted 
up to the 20 following days. 

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

## Moving forward

Once you feel at home with your project, we recommend that you replace
this README file with your own project documentation! Happy data wrangling!
