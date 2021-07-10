import sys, os, pytest
sys.path.append("../")
import numpy as np
from src.utils.data import load_obj, save_obj


@pytest.fixture()
def sequence():
    in_sequence = load_obj("tests/fixture_data/test_inputs.pkl")
    return in_sequence

@pytest.fixture()
def predictions_ref():
    predictions_ref = load_obj("tests/fixture_data/test_pred.pkl")
    return predictions_ref