from tqdm.auto import tqdm


def _get_folds_from_indexes(indexes, set_name='train_'):
    folds = [int(fold.split('_')[-1]) for fold in indexes if fold.startswith(set_name)]
    return folds


def _get_train_and_test_sets(data, indexes, fold, test_mode=False):
    if test_mode:
        train_set = data.loc[indexes['playground']]
        test_set = data.loc[indexes[f'test_{fold}']]
    else:
        train_set = data.loc[indexes[f'train_{fold}']]
        test_set = data.loc[indexes[f'dev_{fold}']]
    return train_set, test_set


def separate_predictors_and_target(data_set, target_col):
    data_set_x = data_set.drop(columns=target_col).values
    data_set_y = data_set[target_col].values
    return data_set_x, data_set_y


def _get_approaches_functions_from_grid(approaches_grid):
    approaches_functions = {app_name: approaches_grid[app_name]['approach_function'] for app_name in approaches_grid}
    return approaches_functions


def run_experiment(experiment, data, indexes, execution_function):
    # Extract all necessary info from experiment.
    folds = _get_folds_from_indexes(indexes)
    approaches_grid = experiment['approaches']
    approaches_functions = _get_approaches_functions_from_grid(approaches_grid)
    evaluation_function = experiment['evaluation_function']
    evaluation_pars = experiment['evaluation_pars']
    metrics = evaluation_pars['metrics']
    execution_pars = experiment['execution_pars']
    target = execution_pars['target']
    test_mode = execution_pars['test_mode']
    exploration_function = experiment['exploration_function']

    # Initialise parameter space explorer.
    explorer = exploration_function(approaches_grid, folds, metrics)
    pars_folds = explorer.initialise_results()
    n_iterations = explorer.select_executions_left()

    for _ in tqdm(range(n_iterations)):
        i, row = explorer.get_next_point()
        # Extract all necessary info from this row.
        fold = row['fold']
        approach_name = row['approach']
        approach_pars = row['pars']
        approach_function = approaches_functions[approach_name]

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
    model = approach_function(**approach_pars)
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    return predictions
