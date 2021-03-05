"""Common functions that are designed for modev.

"""
from modev import default_pars

approach_key = default_pars.approach_key
dev_key = default_pars.dev_key
executed_key = default_pars.executed_key
fold_key = default_pars.fold_key
id_key = default_pars.id_key
pars_key = default_pars.pars_key
playground_key = default_pars.playground_key
test_key = default_pars.test_key
train_key = default_pars.train_key


def get_metrics_from_results(results):
    non_metrics_columns = [pars_key, approach_key, id_key, fold_key, executed_key]
    metrics = [col for col in results.columns if col not in non_metrics_columns]
    return metrics


def get_train_and_test_sets(data, train_indexes, test_indexes, fold):
    # TODO: Instead of this, in test_mode, repeat playground so that train and test sets have the same number of keys.
    if len(train_indexes) == 1:
        train_set = data.loc[train_indexes[0]]
    else:
        train_set = data.loc[train_indexes[fold]]
    test_set = data.loc[test_indexes[fold]]
    return train_set, test_set


def separate_predictors_and_target(data_set, target_col):
    data_set_x = data_set.drop(columns=target_col)
    data_set_y = data_set[target_col].values
    return data_set_x, data_set_y
