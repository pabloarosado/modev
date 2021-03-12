"""Functions related to evaluation metrics.

"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from modev import default_pars
from modev import utils

# List of kwargs accepted by precision and recall functions from sklearn, and accuracy.
precision_recall_f1_kwargs = ['labels', 'pos_label', 'average', 'sample_weight', 'zero_division']
accuracy_kwargs = ['normalize', 'sample_weight']


def prepare_true_and_pred(raw_true, raw_pred):
    """Prepare input ground truth and predictions to have appropriate formats.

    Parameters
    ----------
    raw_true : pd.DataFrame or np.array or list
        Ground truth.
    raw_pred : pd.DataFrame or np.array or list
        Predictions.
    Returns
    -------
    true, pred : np.array
        Ground truth and predictions, in suitable formats.

    """
    # TODO: Ensure they always have the same format, i.e. np.array.
    return raw_true, raw_pred


def evaluate_predictions(execution_results, metrics, **kwargs):
    """Evaluate predictions, given ground truth, using a list of metrics.

    Parameters
    ----------
    execution_results : dict
        Execution results as returned by execution inputs function. It must contain a 'truth' and a 'prediction' key.
    metrics : list
        Metrics to use for evaluation. Implemented methods include:
         * 'precision': usual precision in classification problems.
         * 'recall': usual recall in classification problems.
         * 'f1': usual f1-score in classification problems.
         * 'accuracy': usual accuracy in classification problems.
         * 'precision_at_*': precision at k (e.g. 'precision_at_10') or at k percent (e.g. 'precision_at_5_pct').
         * 'recall_at_*': recall at k (e.g. 'recall_at_10') or at k percent (e.g. 'recall_at_5_pct').
         * 'threshold_at_*': threshold at k (e.g. 'threshold_at_10') or at k percent (e.g. 'threshold_at_5_pct').
        Note: For the time being, all metrics have to return only one number; In the case of a multi-class
        classification, a micro-average precision is returned.

    Returns
    -------
    results : dict
        Results of evaluation. Each element in the dictionary corresponds to one of the metrics.

    """
    raw_true, raw_pred = execution_results[default_pars.truth_key], execution_results[default_pars.prediction_key]
    true, pred = prepare_true_and_pred(raw_true, raw_pred)

    results = {}
    for metric in metrics:
        if metric == 'accuracy':
            usable_kwargs = utils.get_usable_args_for_function(accuracy_score, kwargs, accuracy_kwargs)
            results[metric] = accuracy_score(true, pred, **usable_kwargs)
        elif metric == 'precision':
            usable_kwargs = utils.get_usable_args_for_function(precision_score, kwargs, precision_recall_f1_kwargs)
            results[metric] = precision_score(true, pred, **usable_kwargs)
        elif metric == 'recall':
            usable_kwargs = utils.get_usable_args_for_function(recall_score, kwargs, precision_recall_f1_kwargs)
            results[metric] = recall_score(true, pred, **usable_kwargs)
        elif metric == 'f1':
            usable_kwargs = utils.get_usable_args_for_function(f1_score, kwargs, precision_recall_f1_kwargs)
            results[metric] = f1_score(true, pred, **usable_kwargs)
        elif metric.startswith(('precision_at_', 'recall_at', 'threshold_at_')):
            # Get metrics at k or metrics at k percent (either precision, recall, or threshold).
            k = get_k_from_metric_name(metric, len(pred))
            type_of_metric = metric.split('_at_')[0]
            # Extract only the type of metric at k needed.
            results[metric] = metrics_at_k(true, pred, k)[type_of_metric]
    # TODO: Allow saving metrics like precision and recall as lists (for different labels). Maybe the easiest is to
    #  create metrics *_per_label, that repeat that metric for each of the labels. But for that ensure that 'metrics'
    #  doesn't need to be redefined in pipeline.

    # TODO: Allow saving file with individual predictions in test examples.
    return results


def metrics_at_k(raw_true, raw_pred, k):
    """Calculate metrics at k (e.g. precision@k).

    Parameters
    ----------
    raw_true : np.array
        Ground truth.
    raw_pred : np.array
        Predictions (either booleans, labels, or probabilities, depending on the metric).
    k : int
        Value of k.

    Returns
    -------
    results : dict
        Results of metrics at k, namely 'precision', 'recall', and corresponding 'threshold'.

    """
    true, pred = prepare_true_and_pred(raw_true, raw_pred)
    sorted_indexes = pred.argsort()[::-1]
    true_selected = true[sorted_indexes][0:k]
    pred_selected = pred[sorted_indexes][0:k]
    true_selected_positive = len(true_selected[true_selected])
    true_positive = len(true[true])
    recall_at_k = true_selected_positive / true_positive
    precision_at_k = true_selected_positive / k
    threshold_at_k = pred_selected[-1]
    results = {'precision': precision_at_k, 'recall': recall_at_k, 'threshold': threshold_at_k}
    return results


def get_k_from_metric_name(metric_name, num_predictions=default_pars.evaluation_pars_num_predictions):
    """Get value of k from the name of an "@k" metric.
     If metric is, e.g. 'precision_at_10', this function returns 10.
     If metric is, e.g. 'precision_at_5_pct', this function returns 5% of 'num_predictions' (as integer).

    Parameters
    ----------
    metric_name : str
        Metric name (e.g. 'precision_at_10', or 'precision_at_5_percent')
    num_predictions : int
        Length of predictions array.

    Returns
    -------
    k : int
        Value of k.

    """
    k_or_k_percent = metric_name.split('recall_at_')[-1].split('precision_at_')[-1]
    if k_or_k_percent.isdigit():
        k = int(k_or_k_percent)
    else:
        k_percent = int(k_or_k_percent.split('_')[0])
        k = int(num_predictions * k_percent / 100)
    return k
