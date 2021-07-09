import argparse, sys, os
from utils import data as d_u
from covid_daily.constants import AVAILABLE_COUNTRIES 
import subprocess


def main(folder_name, add_dataset=True):
    """
    main routine, go through each considered availabler countries and try to records corresponding data
    Args:
        folder_name: (str) path to the folder to record data in
        add_dataset: (bool) if True, commit data changes
    """
    success, failures = [], []
    for country in AVAILABLE_COUNTRIES:
        try:
            success.append(d_u.records_country(country, folder_name))
        except:
            failures.append(country)
            print("cannot records country:{0:s}".format(country))
    d_u.records_details(folder_name, {"success": success, "failed_country": failures})
    if add_dataset:
        cmds = ["renku dataset add --overwrite worldometers-data {0:s}".format(" ".join(success))]
        cmds.append("renku dataset add --overwrite worldometers-data {0:s}".format(
            os.path.join(folder_name, "details.json")))
        [subprocess.run(cmd, shell=True) for cmd in cmds]
    return

# main routine
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='fetch data from https://worldometer.info and records csv file for each available country')
    parser.add_argument('-f', '--folder', 
                        help='folder to record the csv in', 
                        type=str, 
                        default="/work/test-first-project/data/worldometers-data/")

    args = parser.parse_args()
    main(folder_name=args.folder)