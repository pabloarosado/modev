import logging
import os
import pickle
import random

import pandas as pd

from modev import utils


def load_experiment_module(experiment_file):
    if not os.path.isfile(experiment_file):
        logging.error("Experiment file not found: %s", experiment_file)
    experiment_module = utils.import_file_as_module(experiment_file, module_name='experiment')
    return experiment_module


def load_experiment(experiment_module):
    if 'experiment' not in dir(experiment_module):
        logging.error("Experiment file must contain a dictionary called 'experiment'.")
    experiment = experiment_module.experiment
    return experiment


def apply_selection_to_data(data, selection):
    sel = eval(selection)
    selected_data = data[sel].copy()
    logging.info("Applying selection: %i rows (of %i) selected.", len(data), len(selected_data))
    return selected_data


def count_rows(data_file):
    with open(data_file) as f:
        num_rows = sum(1 for line in f)
    return num_rows


def sample_from_big_file(data_file, sample_nrows, random_state=1000, header_nrows=1, **kwargs):
    total_nrows = count_rows(data_file)
    random.seed(random_state)
    indexes_to_skip = random.sample(range(header_nrows, total_nrows + header_nrows + 1),
                                    total_nrows + header_nrows - sample_nrows)
    data = pd.read_csv(data_file, skiprows=indexes_to_skip, **kwargs)
    return data


def load_local_file(data_file, selection=None, sample_nrows=None, random_state=1000, **kwargs):
    if not os.path.isfile(data_file):
        logging.error("Data file not found: %s", data_file)
    # Get default args for pd.read_csv.
    usable_kwargs = utils.get_usable_kwargs_for_function(pd.read_csv, kwargs)
    logging.info("Loading data from file %s", data_file)
    data = pd.read_csv(data_file, **usable_kwargs)
    if selection is not None:
        # Create a new dataframe (copy) that fulfils selection, while keeping the original indexes (no reset).
        data = apply_selection_to_data(data, selection)
    if sample_nrows is not None:
        data = data.sample(sample_nrows, random_state=random_state)
    return data


def save_model(model, model_file):
    models_dir = os.path.dirname(model_file)
    if not os.path.isdir(models_dir):
        logging.info("Creating folder for output models: %s", models_dir)
        os.makedirs(models_dir)
    with open(model_file, 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)
