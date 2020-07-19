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


def run_experiment(raw_experiment, data, indexes):
    done_experiment = {}
    approaches = raw_experiment['approaches']
    folds = get_train_folds_from_indexes(indexes)
    evaluator = raw_experiment['evaluation_function']
    target_col = 'category'

    for i, fold in enumerate(folds):
        train_set, test_set = get_train_and_test_sets(data, indexes, fold)
        train_x, train_y = separate_predictors_and_target(train_set, target_col)
        test_x, test_y = separate_predictors_and_target(test_set, target_col)
        for approach_name in approaches:
            approach = approaches[approach_name]
            model = approach['approach_function'](**approach['approach_pars'])
            model.fit(train_x, train_y)
            predictions = model.predict(test_x)
            results = evaluator(list(test_y), list(predictions), **raw_experiment['evaluation_pars'])
            print(results)
    # TODO: store results of different folds in a dataframe.
    return done_experiment
