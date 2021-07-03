import argparse, sys
sys.path.append("/work/test-first-project/src")
import data_utils as data_utils
from covid_daily.constants import AVAILABLE_COUNTRIES 


def main(folder_name):
    """
    main routine, go through each considered availabler countries and try to records corresponding data
    Args:
        folder_name: (str) path to the folder to record data in
    """
    for country in AVAILABLE_COUNTRIES:
        try:
            data_utils.records_country(country, folder_name)
        except:
            print("cannot records country:{0:s}".format(country))
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