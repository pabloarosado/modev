import json
import logging
import os
import pickle

import pandas as pd

from modev import utils


def load_raw_experiment(raw_experiment_file):
    if not os.path.isfile(raw_experiment_file):
        logging.error("Raw experiment file not found: %s", raw_experiment_file)
    raw_experiment = json.load(open(raw_experiment_file, "r"))
    return raw_experiment


def apply_selection_to_data(data, selection):
    sel = eval(selection)
    selected_data = data[sel].copy()
    logging.info("Applying selection: %i rows (of %i) selected.", len(data), len(selected_data))
    return selected_data


def load_local_file(data_file, selection=None, **kwargs):
    if not os.path.isfile(data_file):
        logging.error("Data file not found: %s", data_file)
    # Get default args for pd.read_csv.
    usable_kwargs = utils.get_usable_kwargs_for_function(pd.read_csv, kwargs)
    logging.info("Loading data from file %s", data_file)
    data = pd.read_csv(data_file, **usable_kwargs)
    if selection is not None:
        # Create a new dataframe (copy) that fulfils selection, while keeping the original indexes (no reset).
        data = apply_selection_to_data(data, selection)
    return data


def save_model(model, model_file):
    models_dir = os.path.dirname(model_file)
    if not os.path.isdir(models_dir):
        logging.info("Creating folder for output models: %s", models_dir)
        os.makedirs(models_dir)
    with open(model_file, 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)
