import os
import pandas as pd
import subprocess
import covid_daily
from covid_daily.constants import AVAILABLE_CHARTS 


def records_country(country_name, data_folder, add_dataset=True):
    """
    Get and records data from Worldometers.info
    Args: 
        country_name: (str) name of the country to records data,
        data_folder: (str) folder path to records data in,
        add_dataset: (bool) if True, trigger renku dataset add command
    """
    data = [covid_daily.data(country=country_name, chart=chart, as_json=False) for chart in AVAILABLE_CHARTS]
    data = pd.concat(data, axis=1)
    csv_path = os.path.join(data_folder,"{0:s}.csv".format(country_name))
    data.to_csv(csv_path)
    if add_dataset:
        cmd = "renku dataset add worldometers-data {0:s}".format(csv_path)
        subprocess.run(cmd, shell=True)
    return