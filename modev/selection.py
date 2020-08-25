"""Functions related to model selection.

"""
import numpy as np

from modev import common
from modev import default_pars

approach_key = default_pars.approach_key
id_key = default_pars.id_key
pars_key = default_pars.pars_key


def combine_fold_results(results, aggregation_method=default_pars.selection_pars_aggregation_method):
    # Get metric names from results' columns.
    metrics = common.get_metrics_from_results(results)

    # Combine results for all folds using a certain aggregation method (e.g. mean).
    metrics_agg = {col: aggregation_method for col in metrics}

    # For columns that do not need to be combined, simply take first (since they are identical for all folds).
    other_columns = [approach_key, pars_key]
    other_columns_agg = {col: 'first' for col in other_columns}
    metrics_agg.update(other_columns_agg)
    combined_results = results.groupby(id_key).agg(metrics_agg)
    return combined_results


def rank_models(combined_results, main_metric):
    sorted_results = combined_results.sort_values(main_metric, ascending=False)
    return sorted_results


def apply_condition_to_dataframe(df, condition=default_pars.selection_pars_condition):
    selection = np.ones(len(df), dtype=bool)
    if condition is not None:
        selection = eval(condition)
    df_selected = df[selection].copy()
    return df_selected


def model_selection(results, main_metric, aggregation_method=default_pars.selection_pars_aggregation_method,
                    results_condition=default_pars.selection_pars_results_condition,
                    combined_results_condition=default_pars.selection_pars_combined_results_condition):
    """Model selection.

    Take the evaluation of approaches on some folds, and select the best model.

    Parameters
    ----------
    results : pd.DataFrame
        Evaluations of the performance of approaches on different data folds.
    main_metric : str
        Name of the main metric (the one that has to be maximized).
    aggregation_method : str
        Aggregation method to use to combine evaluations of different folds (e.g. 'mean').
    results_condition : str
        Condition to be applied to results dataframe before combining results from different folds.
    combined_results_condition : str
        Condition to be applied to results dataframe after combining results from different folds.

    Returns
    -------
    combine_results_sorted : pd.DataFrame
        Ranking of results (sorted in descending value of 'main_metric') of approaches that fulfil the imposed
        conditions.

    """
    # Apply conditions to results of individual folds.
    results_selected = apply_condition_to_dataframe(results, results_condition)
    # Combine results of different folds.
    combined_results = combine_fold_results(results_selected, aggregation_method=aggregation_method)
    # Apply conditions to combined results.
    combined_results_selected = apply_condition_to_dataframe(combined_results, combined_results_condition)
    # Create ranking.
    combined_results_sorted = rank_models(combined_results_selected, main_metric=main_metric)
    return combined_results_sorted
