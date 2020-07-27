"""Functions related to the execution of the pipeline.

"""
from tqdm.auto import tqdm

from modev import default_pars


def _get_train_and_test_sets(data, indexes, fold, test_mode=default_pars.execution_pars_test_mode,
                             train_name=default_pars.train_key,
                             dev_name=default_pars.dev_key,
                             test_name=default_pars.test_key,
                             playground_name=default_pars.playground_key):
    if test_mode:
        train_set = data.loc[indexes[playground_name]]
        test_set = data.loc[indexes[f'{test_name}_{fold}']]
    else:
        train_set = data.loc[indexes[f'{train_name}_{fold}']]
        test_set = data.loc[indexes[f'{dev_name}_{fold}']]
    return train_set, test_set


def separate_predictors_and_target(data_set, target_col):
    data_set_x = data_set.drop(columns=target_col).values
    data_set_y = data_set[target_col].values
    return data_set_x, data_set_y


def _get_approaches_functions_from_grid(approaches_grid, function_key=default_pars.function_key):
    approaches_functions = {app_name: approaches_grid[app_name][function_key] for app_name in approaches_grid}
    return approaches_functions


def run_experiment(data, indexes, validation_pars, execution_function, execution_pars, evaluation_function,
                   evaluation_pars, exploration_function, approaches_function, approaches_pars,
                   fold_key=default_pars.fold_key,
                   pars_key=default_pars.pars_key,
                   approach_key=default_pars.approach_key):
    # Extract all necessary info from experiment.
    metrics = evaluation_pars['metrics']
    target = execution_pars['target']
    test_mode = execution_pars['test_mode']

    # Get list of folds to execute.
    if test_mode:
        folds = list(range(validation_pars['test_n_sets']))
    else:
        folds = list(range(validation_pars['playground_n_folds']))

    # Initialise parameter space explorer.
    explorer = exploration_function(approaches_pars, folds, metrics)
    pars_folds = explorer.initialise_results()
    n_iterations = explorer.select_executions_left()

    for _ in tqdm(range(n_iterations)):
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
        results = evaluation_function(list(test_y), list(predictions), **evaluation_pars)

        # Write results for these parameters and fold.
        for metric in results:
            pars_folds.loc[i, metric] = results[metric]
    return pars_folds


def execute_model(approach_function, approach_pars, train_x, train_y, test_x, **kwargs):
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
