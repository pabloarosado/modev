"""Functions related to extraction, transformation and loading (ETL).

"""
import logging
import os
import pickle
import random

import pandas as pd

from modev import default_pars
from modev import utils


def apply_selection_to_data(data, selection):
    sel = eval(selection)
    selected_data = data[sel].copy()
    logging.info("Applying selection: %i rows (of %i) selected.", len(data), len(selected_data))
    return selected_data


def count_rows(data_file):
    with open(data_file) as f:
        num_rows = sum(1 for _ in f)
    return num_rows


def sample_from_big_file(data_file, sample_nrows, random_state=default_pars.random_state,
                         header_nrows=default_pars.etl_pars_header_nrows, **kwargs):
    total_nrows = count_rows(data_file)
    random.seed(random_state)
    indexes_to_skip = random.sample(range(header_nrows, total_nrows + header_nrows + 1),
                                    total_nrows + header_nrows - sample_nrows)
    data = pd.read_csv(data_file, skiprows=indexes_to_skip, **kwargs)
    return data


def load_local_file(data_file, selection=default_pars.etl_pars_selection,
                    sample_nrows=default_pars.etl_pars_sample_nrows, random_state=default_pars.random_state, **kwargs):
    """Load local (.csv) file.

    This function uses pandas.read_csv() function and accepts all its arguments. But it also has some added arguments.

    Parameters
    ----------
    data_file : str
        Path to local (.csv) file.
    selection : str or None
        Selection to perform on the data. For example, if selection is "(data['height'] > 3) & (data['width'] < 2)",
        that selection will be evaluated and applied to the data; None to apply no selection.
    sample_nrows : int or None
        Number of random rows to sample from the data (without repeating rows); None to load all rows.
    random_state : int
        Random state (relevant only when sampling from data, i.e. when 'sample_nrows' is not None).

    Returns
    -------
    data : pd.DataFrame
        Data extracted from 'data_file'.

    """
    if not os.path.isfile(data_file):
        logging.error("Data file not found: %s", data_file)
    # Get default args for pd.read_csv.
    usable_kwargs = utils.get_usable_args_for_function(pd.read_csv, kwargs)
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
