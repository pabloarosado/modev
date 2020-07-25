import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold


def k_folds_split(raw_indexes, n_splits, labels=None, shuffle=True, random_state=1000, return_original_indexes=True):
    raw_indexes = np.array(raw_indexes)
    # Split a data set into n parts without overlap, and optionally stratified.
    if labels is None:
        split_method = KFold
    else:
        split_method = StratifiedKFold
    parts = list(split_method(n_splits=n_splits, random_state=random_state, shuffle=shuffle).
                 split(raw_indexes, labels))
    if return_original_indexes:
        parts = [(raw_indexes[part[0]], raw_indexes[part[1]]) for part in parts]
    return parts


def train_n_tests_split(raw_indexes, test_fraction, test_n_sets=1, labels=None, shuffle=True, random_state=1000,
                        train_name='train', test_name='test'):

    # To begin with, the raw dataset is train, and there is only one test set (named 'test_0'), which is empty.
    indexes = {train_name: np.array(raw_indexes),
               f'{test_name}_0': np.array([], dtype=int)}

    # If 'test_fraction' is not zero, take that fraction from train.
    # The new train will be a fraction 1 - 'test_fraction' of the raw dataset.
    # If labels are given, the splitting will be stratified (otherwise random).
    if test_fraction > 0:
        primary_split = train_test_split(indexes[train_name], test_size=test_fraction, stratify=labels,
                                         random_state=random_state, shuffle=shuffle)
        indexes[train_name] = primary_split[0]
        indexes[f'{test_name}_0'] = primary_split[1]

        # If 'test_n_sets' > 1, we split the test set into 'test_n_sets' sets (named 'test_0', 'test_1', etc.) of
        # approximately equal size.
        # Again, if labels are given, the splitting will be stratified (otherwise random).
        # For convenience, use 'k_folds_split' for this task (and then ignore train parts).
        if test_n_sets > 1:
            test_split = k_folds_split(indexes[f'{test_name}_0'], test_n_sets, labels=labels, shuffle=shuffle,
                                       random_state=random_state)
            # Disregard the zeroth part (which is meant for training), and keep the non-overlapping part.
            indexes.update({f'{test_name}_{i}': fold[1] for i, fold in enumerate(test_split)})
    return indexes


def k_fold_playground_n_tests_split(raw_indexes, playground_n_folds=4, test_fraction=0.2, test_n_sets=1, labels=None,
                                    shuffle=True, random_state=1000):
    # Split data set into playground and test set(s).
    indexes = train_n_tests_split(raw_indexes=raw_indexes, test_fraction=test_fraction, test_n_sets=test_n_sets,
                                  labels=labels, shuffle=shuffle, random_state=random_state, train_name='playground',
                                  test_name='test')
    # Split playground into k train and k dev sets.
    playground_split = k_folds_split(indexes['playground'], playground_n_folds, labels=None, shuffle=True,
                                     random_state=random_state)
    indexes.update({f'train_{i}': part[0] for i, part in enumerate(playground_split)})
    indexes.update({f'dev_{i}': part[1] for i, part in enumerate(playground_split)})
    return indexes


def validate_indexes(indexes):
    # For convenience, collect all indexes in lists.
    train_indexes = []
    dev_indexes = []
    test_indexes = []
    for key in indexes:
        if key.startswith('train_'):
            train_indexes.extend(indexes[key])
        elif key.startswith('dev_'):
            dev_indexes.extend(indexes[key])
        elif key.startswith('test_'):
            test_indexes.extend(indexes[key])

    # Since there can be repetition among indexes in train sets, take unique.
    train_indexes = np.unique(train_indexes)

    # Validations that need to be passed:
    validations = []

    # Playground and train indexes coincide.
    validations.append(set(indexes['playground']) == set(train_indexes))

    # Train indexes and test indexes do not overlap.
    validations.append((set(train_indexes) & set(test_indexes)) == set())

    # For each of the folds in playground, train and dev do not overlap.
    folds = [int(key.split('_')[-1]) for key in indexes if key.startswith('dev_')]
    for fold in folds:
        validations.append(((set(indexes[f'train_{fold}']) & set(indexes[f'dev_{fold}'])) == set()))

    # There is no overlap among dev sets.
    validations.append(len(dev_indexes) == len(np.unique(dev_indexes)))

    # There is no overlap among test sets.
    validations.append(len(test_indexes) == len(np.unique(test_indexes)))

    check_validations = np.sum(validations) == len(validations)
    return check_validations
