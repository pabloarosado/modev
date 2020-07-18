from sklearn.model_selection import train_test_split, StratifiedKFold, KFold


def folding(raw_indexes, labels=None, folding_type=None, test_size=None, n_folds=4, stratified=True, random_state=1000):
    # TODO: start implementing k-folding CV.
    # TODO: then implement temporal folding.

    indexes = {}
    if test_size is None:
        playground_indexes = raw_indexes
    else:
        if labels is not None and stratified:
            splits = train_test_split(raw_indexes, random_state=random_state, shuffle=True, test_size=test_size,
                                      stratify=labels)
        else:
            splits = train_test_split(raw_indexes, random_state=random_state, shuffle=True, test_size=test_size)
        playground_indexes = splits[0]
        test_indexes = splits[1]

    if (folding_type == 'k_fold') or folding_type is None:
        if stratified and labels is not None:
            folds = list(StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=True).
                         split(playground_indexes, labels))
        else:
            folds = list(KFold(n_splits=n_folds, random_state=random_state, shuffle=True).
                         split(playground_indexes))

    # TODO: now return indexes dictionary.