import sklearn.metrics


def prepare_true_and_pred(true, pred):
    # TODO: Ensure they always have the same format.
    return true, pred


def evaluate_predictions(raw_true, raw_pred, metrics):
    true, pred = prepare_true_and_pred(raw_true, raw_pred)
    results = {}
    if 'accuracy' in metrics:
        results['accuracy'] = sklearn.metrics.accuracy_score(true, pred)
    if 'precision' in metrics:
        results['accuracy'] = sklearn.metrics.precision_score(true, pred)
    return results
