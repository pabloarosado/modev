"""Functions related to the execution of the pipeline.

"""
import os

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from modev import default_pars

approach_key = default_pars.approach_key
dev_key = default_pars.dev_key
fold_key = default_pars.fold_key
function_key = default_pars.function_key
pars_key = default_pars.pars_key
playground_key = default_pars.playground_key
test_key = default_pars.test_key
train_key = default_pars.train_key


def _get_train_and_test_sets(data, indexes, fold, test_mode=default_pars.execution_pars_test_mode):
    if test_mode:
        train_set = data.loc[indexes[playground_key]]
        test_set = data.loc[indexes[f'{test_key}_{fold}']]
    else:
        train_set = data.loc[indexes[f'{train_key}_{fold}']]
        test_set = data.loc[indexes[f'{dev_key}_{fold}']]
    return train_set, test_set


def separate_predictors_and_target(data_set, target_col):
    data_set_x = data_set.drop(columns=target_col)
    data_set_y = data_set[target_col].values
    return data_set_x, data_set_y


def _get_approaches_functions_from_grid(approaches_grid):
    approaches_functions = {app_name: approaches_grid[app_name][function_key] for app_name in approaches_grid}
    return approaches_functions


def _add_metrics_to_pars_folds(i, pars_folds, results):
    for metric in results:
        if metric not in pars_folds.columns:
            pars_folds[metric] = np.nan
        pars_folds.loc[i, metric] = results[metric]


def _get_list_of_sets_from_indexes(indexes, set_name):
    list_of_sets = [int(part.split('_')[-1]) for part in indexes if part.startswith(set_name)]
    return list_of_sets


def run_experiment(data, indexes, execution_function, execution_pars, evaluation_function, evaluation_pars,
                   exploration_function, approaches_function, approaches_pars, results_file=None,
                   save_every=default_pars.save_every, reload=False):
    # Extract all necessary info from experiment.
    target = execution_pars['target']
    test_mode = execution_pars['test_mode']

    # Get list of folds to execute.
    if test_mode:
        folds = _get_list_of_sets_from_indexes(indexes, test_key)
    else:
        folds = _get_list_of_sets_from_indexes(indexes, dev_key)

    # Optionally (if an existing results_file is given) load results from file; if reload is True, ignore that file.
    if (results_file is not None) and os.path.isfile(results_file) and not reload:
        pars_folds = pd.read_csv(results_file)
        pars_folds[default_pars.pars_key] = [eval(row) for row in pars_folds[default_pars.pars_key]]
    else:
        pars_folds = None

    # Initialise parameter space explorer.
    explorer = exploration_function(approaches_pars, folds, pars_folds)
    pars_folds = explorer.initialise_results()
    n_iterations = explorer.select_executions_left()

    for iteration in tqdm(range(n_iterations)):
        i, row = explorer.get_next_point()
        # Extract all necessary info from this row.
        fold = row[fold_key]
        approach_name = row[approach_key]
        approach_pars = row[pars_key]
        approach_function = approaches_function[approach_name]

        # Get train_* and test_* sets.
        train_set, test_set = _get_train_and_test_sets(data, indexes, fold, test_mode)
        train_x, train_y = separate_predictors_and_target(train_set, target)
        test_x, test_y = separate_predictors_and_target(test_set, target)

        # Fit and predict with approach.
        predictions = execution_function(approach_function, approach_pars, train_x, train_y, test_x, **execution_pars)

        # Evaluate predictions.
        results = evaluation_function(test_y, predictions, **evaluation_pars)

        # Ensure metrics columns exist in pars_folds and write results for these parameters and fold.
        _add_metrics_to_pars_folds(i, pars_folds, results)

        # Mark current row as executed.
        pars_folds.loc[i, default_pars.executed_key] = True

        # Optionally save temporary results to file.
        if results_file is not None and ((iteration + 1) % save_every == 0):
            pars_folds.to_csv(results_file, index=False)

    # Optionally save finished results to file.
    if results_file is not None:
        pars_folds.to_csv(results_file, index=False)
    return pars_folds


def execute_model(approach_function, approach_pars, train_x, train_y, test_x, **_kwargs):
    """Execution method (including training and prediction) for an approach.

    This function takes an approach 'approach_function' with parameters 'approach_pars', a train set (with predictors
    'train_x' and targets 'train_y') and the predictors of a test set 'test_x', and returns the predicted targets of the
    test set.
    Note: Here, 'test' refers to either a dev or a test set indistinctly.

    Parameters
    ----------
    approach_function : class
        Approach that contains a 'fit' method (to fit approach on train set) and a 'predict' method (to predict on the
        test set).
    approach_pars : dict
        Parameters for the approach.
    train_x : array_like
        Predictors for the train set.
    train_y : array_like
        Target values for the train set.
    test_x : array_like
        Predictors for the test set (or dev set).

    Returns
    -------
    predictions: array_like
        Predictions for the test set (or dev set).

    """
    model = approach_function(**approach_pars)
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    return predictions
