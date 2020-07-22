import numpy as np


def combine_fold_results(results, metrics, aggregation_method='mean'):
    # Combine results for all folds using a certain aggregation method (e.g. mean).
    metrics_agg = {col: aggregation_method for col in metrics}
    # For columns that do not need to be combined, simply take first (since they are identical for all folds).
    # other_columns = [col for col in pipe.results.columns if col not in metrics]
    other_columns = ['approach', 'pars']
    other_columns_agg = {col: 'first' for col in other_columns}
    metrics_agg.update(other_columns_agg)
    combined_results = results.groupby('id').agg(metrics_agg)
    return combined_results


def rank_models(combined_results, main_metric):
    sorted_results = combined_results.sort_values(main_metric, ascending=False)
    return sorted_results


def apply_condition_to_dataframe(df, condition=None):
    selection = np.ones(len(df), dtype=bool)
    if condition is not None:
        selection = eval(condition)
        # TODO: Here ensure the condition is well structured.
    df_selected = df[selection].copy()
    return df_selected


def model_selection(results, metrics, main_metric, aggregation_method='mean', results_condition=None,
                    combined_results_condition=None):
    # Apply conditions to results of individual folds.
    results_selected = apply_condition_to_dataframe(results, results_condition)
    # Combine results of different folds.
    combined_results = combine_fold_results(results_selected, metrics, aggregation_method=aggregation_method)
    # Apply conditions to combined results.
    combined_results_selected = apply_condition_to_dataframe(combined_results, combined_results_condition)
    # Create ranking.
    combined_results_sorted = rank_models(combined_results_selected, main_metric=main_metric)
    return combined_results_sorted
