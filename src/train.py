import argparse, sys, os
from utils.data import load_training_data, reads_details, records_details
from utils import train as t_u
import numpy as np
import subprocess


def main(model_name, n_epochs=30, n_units=50, models_folder="/work/test-first-project/data/models/", add_dataset=True,
        shuffle=False):
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
    np.random.seed(0)
    input_seq, output_seq, scaler = t_u.scale_data(input_seq, output_seq)
    in_timesteps, out_timesteps, n_features = input_seq.shape[1], output_seq.shape[1], input_seq.shape[2]
    print("sequences length, inputs: {0:d}, output:{1:d}".format(in_timesteps, out_timesteps))
    
    # set up model, train and record
    model = t_u.set_MVar_EncDec_lstm(in_timesteps, out_timesteps, n_features, n_units=n_units)
    model.fit(input_seq, output_seq, epochs=n_epochs, batch_size=512, validation_split=0.08)
    model_path = os.path.join(models_folder, model_name)
    t_u.save_model_score(model_path, model, scaler=scaler)
    
    # records details
    format_details = reads_details("/work/test-first-project/data/model-data")
    details = {"n_units":[n_units], "in_timesteps":[in_timesteps], 
               "out_timesteps":[out_timesteps], "n_features":[n_features],
               "training_data": {"formatting_date": format_details["processing_date"], "downloading_date": format_details["downloading_date"],
                                "countries":format_details["formated_countries"]}}
    records_details(model_path, details)
    
    # update benchmark
    t_u.record_benchmark_graph()
    # track dataset records
    if add_dataset:
        cmd = "renku dataset add --overwrite models {0:s}".format(model_path)
        subprocess.run(cmd, shell=True)
    return

# main routine
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='fetch data from https://worldometer.info and records csv file for each available country')
    parser.add_argument('-m', '--model_name', 
                        help='backup name of the model', 
                        type=str, 
                        default="model_3")
    parser.add_argument('-e', '--epochs', 
                        help='number of epochs to train', 
                        type=int, 
                        default=30)
    parser.add_argument('-u', '--n_units', 
                        help='number of units in the lstm layer composing the model', 
                        type=int, 
                        default=50)
    parser.add_argument('-t', '--track_data_change', 
                        help='if True, add and push dataset change (default True)', 
                        type=lambda x: (str(x).lower() == 'true'), 
                        default=True)

    args = parser.parse_args()
    main(model_name=args.model_name, n_epochs=args.epochs, n_units=args.n_units, add_dataset=args.track_data_change)