from tqdm.auto import tqdm


def get_train_folds_from_indexes(indexes):
    folds = [int(fold.split('_')[-1]) for fold in indexes if fold.startswith('train_')]
    return folds


def get_train_and_test_sets(data, indexes, fold, test_mode=False):
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


def run_experiment(experiment, data, indexes):
    results = {}
    folds = get_train_folds_from_indexes(indexes)
    approaches_grid = experiment['approaches']
    approaches_functions = {app_name: approaches_grid[app_name]['approach_function'] for app_name in approaches_grid}
    evaluation_function = experiment['evaluation_function']
    evaluation_pars = experiment['evaluation_pars']
    metrics = evaluation_pars['metrics']
    target = experiment['execution_pars']['target']
    # Initialise explorer.
    explorer = experiment['exploration_function'](approaches_grid, folds, metrics)
    pars_folds = explorer.initialise_results()
    n_iterations = explorer.select_executions_left()

    for _ in tqdm(range(n_iterations)):
        i, row = explorer.get_next_point()
        fold = row['fold']
        approach_name = row['approach']
        approach_pars = row['pars']
        approach_function = approaches_functions[approach_name]

        ground_truth, predictions = execute_predictions(data, indexes, fold, target, approach_function, approach_pars)
        results = evaluation_function(list(ground_truth), list(predictions), **evaluation_pars)
        for metric in results:
            pars_folds.loc[i, metric] = results[metric]
    return pars_folds


def execute_predictions(data, indexes, fold, target, approach_function, approach_pars):
    train_set, test_set = get_train_and_test_sets(data, indexes, fold)
    train_x, train_y = separate_predictors_and_target(train_set, target)
    model = approach_function(**approach_pars)
    model.fit(train_x, train_y)

    test_x, test_y = separate_predictors_and_target(test_set, target)
    predictions = model.predict(test_x)
    return test_y, predictions
