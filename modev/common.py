"""Common functions that are designed for modev.

"""
from modev import default_pars


def get_metrics_from_results(results, pars_key=default_pars.pars_key, approach_key=default_pars.approach_key,
                             id_key=default_pars.id_key, fold_key=default_pars.fold_key):
    non_metrics_columns = [pars_key, approach_key, id_key, fold_key]
    metrics = [col for col in results.columns if col not in non_metrics_columns]
    return metrics
