import argparse, sys, os
sys.path.append("/work/test-first-project/src")
from data_utils import load_training_data
import train_utils as t_u
import numpy as np
import subprocess


def main(model_name, n_epochs=30, n_units=50, models_folder="/work/test-first-project/data/models/", add_dataset=True):
    """
    main routine, load training data, set up, train and records model.
    args:
        model_name: (str) backup name of the model,
        n_epochs: (int) number of epoch to train,
        n_units: (int) number of units in the LSTM layer,
        models_folder: (str) path of the folder to record model in
        add_dataset: (bool) if True, commit data changes
    """
    # load training data and intialize parameters
    input_seq, output_seq, last_in_dates, country_array = load_training_data()
    in_timesteps, out_timesteps, n_features = input_seq.shape[1], output_seq.shape[1], input_seq.shape[2]
    
    # set up model, train and record
    model = t_u.set_MVar_EncDec_lstm(in_timesteps, out_timesteps, n_features, n_units=n_units)
    model.fit(input_seq, output_seq, epochs=n_epochs, batch_size=512, validation_split=0.08)
    model_path = os.path.join(models_folder, model_name)
    t_u.save_model_score(model_path, model)
    
    # track dataset records
    if add_dataset:
        cmd = "renku dataset add --overwrite model-data {0:s}".format(model_path)
        subprocess.run(cmd, shell=True)
    return

# main routine
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='fetch data from https://worldometer.info and records csv file for each available country')
    parser.add_argument('-m', '--model_name', 
                        help='backup name of the model', 
                        type=str, 
                        default="model_1")

    args = parser.parse_args()
    main(model_name=args.model_name)