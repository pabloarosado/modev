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


def get_train_and_test_sets(data, indexes, fold, test_mode):
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
