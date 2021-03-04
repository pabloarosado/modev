"""Common functions that are designed for modev.

"""
from modev import default_pars

approach_key = default_pars.approach_key
id_key = default_pars.id_key
fold_key = default_pars.fold_key
pars_key = default_pars.pars_key
executed_key = default_pars.executed_key


def get_metrics_from_results(results):
    non_metrics_columns = [pars_key, approach_key, id_key, fold_key, executed_key]
    metrics = [col for col in results.columns if col not in non_metrics_columns]
    return metrics
