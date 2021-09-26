import os, argparse, subprocess
import pandas as pd
import numpy as np
from utils.data import load_training_data, save_obj, reads_details, records_details
from utils import train as t_u

def main(model_name, models_folder, inference_folder, add_dataset=True):
    """
    main routine, load saved model, saved sequences to infer, apply model and store predictions.
    args:
        model_name: (str) backup name of the model,
        models_folder: (str) folder containing the saved models,
        inference_folder: (str) folder to solve inference data in
        add_dataset: (bool) if True, commit data changes
    """
    # load model and sequence to infer
    model_folder = os.path.join(models_folder, model_name)
    model = t_u.scaled_model(model_folder)
    input_seq, output_seq, last_in_dates, country_array, _ = load_training_data(as_dataset=False, scale=False, shuffle=False)
    predictions = model.predict(input_seq)

    # record sequences and details
    var_list = [predictions, last_in_dates, country_array]
    name_list = ["predictions.pkl", "last_in_dates", "country.pkl"]
    for var, var_name in zip(var_list, name_list):
        save_obj(var, os.path.join(inference_folder, var_name))
    details_dic = reads_details(model_folder)
    del details_dic["processing_date"]
    details_dic["model"] = model_name
    records_details(inference_folder, details_dic)
    
    # track dataset records
    if add_dataset:
        files = [os.path.join(inference_folder, f) for f in os.listdir(inference_folder)]
        cmd = "renku dataset add --overwrite inference-data {0:s}".format(" ".join(files))
        subprocess.run(cmd, shell=True)
    return

# main routine
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='apply model for inference and store predictions')
    parser.add_argument('-m', '--model_name', 
                        help='name of the model to use', 
                        type=str, 
                        default="model_4")
    parser.add_argument('--models_folder', 
                        help='folder containing the saved models', 
                        type=str, 
                        default="/work/test-first-project/data/models/")
    parser.add_argument('--inference_folder', 
                        help='folder to solve inference data in', 
                        type=str, 
                        default="/work/test-first-project/data/inference-data/")
    parser.add_argument('-t', '--track_data_change', 
                        help='if True, add and push dataset change (default True)', 
                        type=lambda x: (str(x).lower() == 'true'), 
                        default=True)

    args = parser.parse_args()
    model_name = args.model_name
    models_folder = args.models_folder
    inference_folder = args.inference_folder
    track_data_change = args.track_data_change
    main(model_name, models_folder, inference_folder, add_dataset=track_data_change)
   
