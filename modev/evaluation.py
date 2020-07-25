from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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


def evaluate_predictions(raw_true, raw_pred, metrics, **kwargs):
    """Evaluate predictions, given ground truth, using a list of metrics.

    Parameters
    ----------
    raw_true : np.array
        Ground truth.
    raw_pred : np.array
        Predictions (either booleans, labels, or probabilities, depending on the metric).
    metrics : list
        Metrics to use for evaluation (e.g. ['precision', 'recall'])

    Returns
    -------
    results : dict
        Results of evaluation. Each element in the dictionary corresponds to one of the metrics.

    """
    true, pred = prepare_true_and_pred(raw_true, raw_pred)

    results = {}
    for metric in metrics:
        if metric == 'accuracy':
            usable_kwargs = utils.get_usable_args_for_function(accuracy_score, kwargs, accuracy_kwargs)
            results[metric] = accuracy_score(true, pred, **usable_kwargs)
        elif metric == 'precision':
            # TODO: Allow saving metrics like precision and recall as lists (for different labels).
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


def get_k_from_metric_name(metric_name, num_predictions=None):
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
