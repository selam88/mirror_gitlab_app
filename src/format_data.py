import argparse, sys, os
from utils import data as d_u
import numpy as np
import pandas as pd
import subprocess


def main(csv_folder,
         input_variables, output_variable, 
         in_timesteps, out_timesteps, 
         resampling_rule, add_dataset=True):
    """
    main routine, merge csv files, creates and records sequences
    args:
        csv_folder: (str) path to the folder containing recorded csv files,
        input_variables: ([str]) variables to use as input,
        output_variable: (str) variable to use as target to predict,
        in_timesteps: (int) number of input timestep, 
        out_timesteps: (int) number of output timestep,
        resampling_rule: (str) rule to apply resampling ("None"(default), "W", "M", "W-Mon"...)
        add_dataset: (bool) if True, commit data changes
    """
    #parse csv file
    csv_files = [os.path.join(csv_folder, f) for f in os.listdir(csv_folder) if f.endswith(".csv")]
    df_list = []

    # merge available csv
    for file in csv_files:
        country = file.split("/")[-1].split(".")[0]
        df = pd.read_csv(file, parse_dates=True, index_col=0)
        new_column_name = {k: "{0:s}-{1:s}".format(country, k).replace(" ", "_") for k in df.columns}
        df.rename(columns=new_column_name, inplace=True)
        df_list.append(df)
    overall_df = pd.concat(df_list, axis=1)

    # check that no dates are lost
    for df in df_list: 
        assert np.sum(~df.index.isin(overall_df.index))==0

    # if we choose to resample 
    if not isinstance(resampling_rule, type(None)):
        overall_df = overall_df.resample(resampling_rule).sum()

    # create sequences
    for i, country_df in enumerate(df_list): 
        country = country_df.columns[0].split("-")[0]
        nan_free_df = country_df.dropna().rename(columns={k: k.split("-")[1] for k in country_df.columns})
        nan_free_df = nan_free_df[input_variables]
        assert nan_free_df.shape[1]==2
        xx, yy, xx_last_date = d_u.get_multivariate_sequence(nan_free_df, output_variable, in_timesteps, out_timesteps)
        if i==0:
            input_seq, output_seq, last_in_dates = xx, yy, xx_last_date
            country_array = np.array([country for i in range(len(xx))])
            continue
        input_seq = np.concatenate((input_seq, xx), axis=0)
        output_seq =np.concatenate((output_seq, yy), axis=0) 
        last_in_dates = np.concatenate((last_in_dates, xx_last_date), axis=0)
        country_array = np.concatenate((country_array, np.array([country for i in range(len(xx))])), axis=0)
        assert input_seq.shape[1]==in_timesteps
        assert output_seq.shape[1]==out_timesteps
        for v in [output_seq, last_in_dates, country_array]:
            assert v.shape[0]==input_seq.shape[0] 
    output_seq = output_seq[:,:, np.newaxis]
            
    # record sequences
    model_data_folder = "/work/test-first-project/data/model-data/"
    d_u.save_obj(input_seq, os.path.join(model_data_folder, "input_sequences.pkl"))
    d_u.save_obj(output_seq, os.path.join(model_data_folder, "output_sequences.pkl"))
    d_u.save_obj(last_in_dates, os.path.join(model_data_folder, "last_in_dates.pkl"))
    d_u.save_obj(country_array, os.path.join(model_data_folder, "country.pkl"))
            
    # track dataset records
    if add_dataset:
        files = [os.path.join(model_data_folder, f) for f in os.listdir(model_data_folder)]
        cmd = "renku dataset add --overwrite model-data {0:s}".format(" ".join(files))
        subprocess.run(cmd, shell=True)
    return

# main routine
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='format csv data from Worldometer into trainable sequences')
    parser.add_argument('--csv_folder', 
                        help='folder containing recorded the csv', 
                        type=str, 
                        default="/work/test-first-project/data/worldometers-data/")
    parser.add_argument('--input_variables', 
                        help='variables to use as input', 
                        type=str, nargs='+',
                        default=["Currently_Infected", "Novel_Coronavirus_Daily_Cases"])
    parser.add_argument('--output_variable', 
                        help='variable to predict', 
                        type=str, 
                        default="Novel_Coronavirus_Daily_Cases")
    parser.add_argument('--in_timesteps', 
                        help='Number of input timestep', 
                        type=int, 
                        default=20)
    parser.add_argument('--out_timesteps', 
                        help='Number of output timestep', 
                        type=int, 
                        default=20)
    parser.add_argument('--resampling_rule', 
                        help='Rule to apply resampling ("None"(default), "W", "M", "W-Mon"...)', 
                        type=str, 
                        default="None")

    args = parser.parse_args()
    csv_folder = args.csv_folder
    input_variables = args.input_variables
    output_variable = args.output_variable
    in_timesteps = args.in_timesteps
    out_timesteps = args.out_timesteps
    resampling_rule = None if args.resampling_rule=="None" else args.resampling_rule
    main(csv_folder, input_variables, output_variable, in_timesteps, out_timesteps, resampling_rule)
   
