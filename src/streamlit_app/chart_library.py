import os, sys
import pandas as pd
import numpy as np
import streamlit as st 
from datetime import timedelta
import altair as alt
from copy import deepcopy


def get_first_chart(main_vis_df, country, cutoff):
    """
    Create overall country predictions/situation graph
    args:
        main_vis_df: (pandas DataFrame) Dataframe of value to display 
                     with "Daily new cases", "min" and "max" attributes
                     "min" and "max" attribute translate the interval
                     of predictions
        country: (str) Name of the country to display
        cutoo: (pandas Dataframe) Dataframe with cutoff value defining 
                the monthly selected interval
    return:
        altair.chart
    """
    area = alt.Chart(main_vis_df).mark_area(opacity=0.5, color='#cb181d').encode(
        x=alt.X('index', axis=alt.Axis(title='Date')),
        y=alt.Y('Min'), y2=alt.Y2('Max')
    ).properties(width=800, height=350, 
                 title="Situation in {0:s} and corresponding interval of predictions".format(country.title())).interactive()
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
    vertical_line = alt.Chart(pd.DataFrame(data={"vert_line":[date_timest]})).mark_rule(strokeDash=[5,5]).encode(
    x='vert_line', color=alt.ColorValue('#4c78a8'), size=alt.value(1.5))
    return area+line+month_box+vertical_line

def get_second_chart(monthly_error_df, date_timestamp):
    """
    Create violin plots of average monthly errors per timesteps
    args:
        monthly_error_df: (pandas DataFrame) Dataframe of value to display 
                        with "Error" and "Daily timesteps" attributes
        date_timestamp: (pandas Timestamp) Timestamp date of the predictions
    return:
        altair.chart
    """
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
                #labelOrient='bottom',
                labelPadding=0,
            ),
        )
    ).properties(title="Average prediction error per timestep in {0:s} {1:d}".format(date_timestamp.month_name(), date_timestamp.year),
        width=35
    ).configure_title(
    anchor='middle'
    ).configure_facet(
        spacing=0
    ).configure_view(
        stroke=None
    )
    return error_chart

def get_third_chart(multi_seg_df, date_timest):
    """
    Create multi-time serie line graph
    args:
        multi_seg_df: (pandas DataFrame) Dataframe of value to display 
                       with "Date", "Daily new cases" and "Category" attributes
    return:
        altair.chart
    """
    third_chart = alt.Chart(multi_seg_df).mark_line().encode(
        x='Date',
        y='Daily new cases',
        color='Category',
        strokeDash='Category',
    ).interactive()
    vertical_line = alt.Chart(pd.DataFrame(data={"vert_line":[date_timest]})).mark_rule(strokeDash=[5,5]).encode(
    x='vert_line', color=alt.ColorValue('#4c78a8'), size=alt.value(1.5))
    return third_chart + vertical_line
