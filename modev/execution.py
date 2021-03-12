"""Functions related to the execution of the pipeline.

"""
import os

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from modev import common
from modev import default_pars

approach_key = default_pars.approach_key
dev_key = default_pars.dev_key
fold_key = default_pars.fold_key
function_key = default_pars.function_key
pars_key = default_pars.pars_key
playground_key = default_pars.playground_key
test_key = default_pars.test_key
train_key = default_pars.train_key


def _get_approaches_functions_from_grid(approaches_grid):
    approaches_functions = {app_name: approaches_grid[app_name][function_key] for app_name in approaches_grid}
    return approaches_functions


def _add_metrics_to_pars_folds(i, pars_folds, results):
    for metric in results:
        if metric not in pars_folds.columns:
            pars_folds[metric] = np.nan
        pars_folds.loc[i, metric] = results[metric]


def run_experiment(data, train_indexes, test_indexes, execution_function, execution_pars, evaluation_function,
                   evaluation_pars, exploration_function, approaches_function, approaches_pars, results_file=None,
                   save_every=default_pars.save_every, reload=False):
    # Get list of folds to execute.
    folds = list(test_indexes)

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
        model = approach_function(**approach_pars)

        # TODO: In test_mode, repeat playground so that train and test sets always have the same number of keys. Then
        #  remove the following condition.
        if len(train_indexes) == 1:
            fold_train_indexes = train_indexes[0]
        else:
            fold_train_indexes = train_indexes[fold]
        fold_test_indexes = test_indexes[fold]

        # Fit and predict with approach.
        execution_results = execution_function(
            model, data, fold_train_indexes, fold_test_indexes, **execution_pars)

        # Evaluate predictions.
        evaluation_results = evaluation_function(execution_results, **evaluation_pars)

        # Ensure metrics columns exist in pars_folds and write results for these parameters and fold.
        _add_metrics_to_pars_folds(i, pars_folds, evaluation_results)

        # Mark current row as executed.
        pars_folds.loc[i, default_pars.executed_key] = True

        # Optionally save temporary results to file.
        if results_file is not None and ((iteration + 1) % save_every == 0):
            pars_folds.to_csv(results_file, index=False)

    # Optionally save finished results to file.
    if results_file is not None:
        pars_folds.to_csv(results_file, index=False)
    return pars_folds


def execute_model(model, data, fold_train_indexes, fold_test_indexes, target, **_kwargs):
    """Execution method (including training and prediction) for an approach.

    This function takes an approach 'approach_function' with parameters 'approach_pars', a train set (with predictors
    'train_x' and targets 'train_y') and the predictors of a test set 'test_x', and returns the predicted targets of the
    test set.
    Note: Here, 'test' refers to either a dev or a test set indistinctly.

    Parameters
    ----------
    model : model object
        Approach (already initialised with approach parameters) that contains a 'fit' method (to fit approach on train
        set) and a 'predict' method (to predict on the test set).
    data : pd.DataFrame
        Data, as returned by load inputs function.
    fold_train_indexes : np.array
        Indexes of train set (or playground set) for current fold.
    fold_test_indexes : np.array
        Indexes of dev set (or test set) for current fold.
    target : str
        Name of target column in both train_set and test_set.

    Returns
    -------
    execution_results : dict
        Execution results. It contains:
        * 'truth': np.array of true values of the target in the dev (or test) set.
        * 'prediction': np.array of predicted values of the target in the dev (or test) set.

    """
    # A preprocessing method of the model could be applied here:
    # model.preprocess(data)
    # That method could select columns to be used as predictors for train x.

    # Select train set and fit model.
    train_set = data.loc[fold_train_indexes]
    train_x, train_y = common.separate_predictors_and_target(train_set, target)
    model.fit(train_x, train_y)

    # Select test set and predict with model.
    test_set = data.loc[fold_test_indexes]
    test_x, test_y = common.separate_predictors_and_target(test_set, target)
    prediction = model.predict(test_x)

    # Prepare execution results (other metrics like timing could also be included here).
    execution_results = {default_pars.truth_key: test_y, default_pars.prediction_key: prediction}

    return execution_results
