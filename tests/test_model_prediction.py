import sys, os
sys.path.append("../")
import numpy as np
from src.utils.train import *
from .fixtures import sequence, predictions_ref

def test_unscaled_predictions(sequence, predictions_ref):
    model_folder = "tests/fixture_data/models_2/"
    model = scaled_model(model_folder)
    predictions = model.predict(sequence)
    assert (predictions==predictions_ref).all()