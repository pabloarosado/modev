"""Functions related to the validation process, e.g. k-fold or temporal-fold cross-validation.

"""
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

from modev import default_pars


def k_folds_split(raw_indexes, n_splits, labels=default_pars.validation_pars_labels,
                  shuffle=default_pars.validation_pars_shuffle, random_state=default_pars.random_state,
                  return_original_indexes=default_pars.validation_pars_return_original_indexes):
    raw_indexes_array = np.array(raw_indexes)
    # Split a data set into n parts without overlap, and optionally stratified.
    if labels is None:
        split_method = KFold
    else:
        split_method = StratifiedKFold
    parts = list(split_method(n_splits=n_splits, random_state=random_state, shuffle=shuffle).
                 split(raw_indexes_array, labels))
    if return_original_indexes:
        parts = [(raw_indexes_array[part[0]], raw_indexes_array[part[1]]) for part in parts]
    return parts


def temporal_folds_split(raw_indexes, min_n_train_examples, dev_n_sets):
    # Simplest temporal validation: separate the first 'min_n_train_examples' examples for the first train set.
    # Then split the remaining example homogeneously in 'dev_n_sets' sets; they will be the dev sets.
    # The corresponding train set of each of these sets will be made of all previous examples.
    raw_indexes_array = np.array(raw_indexes)
    first_train = raw_indexes_array[0: min_n_train_examples]

    dev_splits = np.array_split(raw_indexes_array[min_n_train_examples:], dev_n_sets)

    parts = [(first_train, dev_splits[0])]
    for fold in range(1, dev_n_sets):
        dev_fold = dev_splits[fold]
        train_fold = raw_indexes_array[raw_indexes_array < dev_fold[0]]
        parts.append((train_fold, dev_fold))
    return parts


def train_n_tests_split(raw_indexes, test_fraction, test_n_sets=default_pars.validation_pars_test_n_sets,
                        labels=default_pars.validation_pars_labels, shuffle=default_pars.validation_pars_shuffle,
                        random_state=default_pars.random_state,
                        train_name=default_pars.train_key,
                        test_name=default_pars.test_key):

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


def k_fold_playground_n_tests_split(raw_indexes, playground_n_folds=default_pars.validation_pars_playground_n_folds,
                                    test_fraction=default_pars.validation_pars_test_fraction,
                                    test_n_sets=default_pars.validation_pars_test_n_sets,
                                    labels=default_pars.validation_pars_labels,
                                    shuffle=default_pars.validation_pars_shuffle,
                                    random_state=default_pars.random_state,
                                    train_name=default_pars.train_key,
                                    dev_name=default_pars.dev_key,
                                    playground_name=default_pars.playground_key,
                                    test_name=default_pars.test_key):
    """Generate indexes that split data into a playground (with k folds) and n test sets.

    There is only one playground, which contains train and dev sets, and has no overlap with test sets.
    Playground is split into k folds, namely k non-overlapping dev sets, and k overlapping train sets.
    Each of the folds contains all data in the playground (part of it in train, and the rest in dev); hence train and
    dev sets of the same fold do not overlap.

    Parameters
    ----------
    raw_indexes : array_like
        All indexes of data. This could simply be the output of 'data.index' (assuming data is a pandas dataframe).
    playground_n_folds : int
        Number of folds to split playground into (also called 'k'), so that there will be k train sets and k dev sets.
    test_fraction : float
        Fraction of data to use for test sets.
    test_n_sets : int
        Number of test sets.
    labels : list or None
        Labels to stratify data according to their distribution; None to not stratify data.
    shuffle : bool
        True to shuffle data before splitting; False to keep them sorted as they are before splitting.
    random_state : int
        Random state to use on the splittings.
    train_name : str
        Name given to the train set (usually 'train').
    dev_name : str
        Name given to the dev set (usually 'dev').
    playground_name : str
        Name given to the playground (usually 'playground').
    test_name : str
        Name given to the test set (usually 'test').

    Returns
    -------
    indexes : dict
        Indexes to use for validation. It contains one playground (named 'playground') and test sets (named 'test_0',
        ..., 'test_n'). The indexes of train and dev sets (contained in the playground) are also given (named 'train_0',
        ..., 'train_k' and 'dev_0', ..., 'dev_k', respectively).

    """
    # Split data set into playground and test set(s).
    indexes = train_n_tests_split(raw_indexes=raw_indexes, test_fraction=test_fraction, test_n_sets=test_n_sets,
                                  labels=labels, shuffle=shuffle, random_state=random_state, train_name=playground_name,
                                  test_name=test_name)
    # Split playground into k train and k dev sets.
    playground_split = k_folds_split(indexes[playground_name], playground_n_folds, labels=None, shuffle=True,
                                     random_state=random_state)
    indexes.update({f'{train_name}_{i}': part[0] for i, part in enumerate(playground_split)})
    indexes.update({f'{dev_name}_{i}': part[1] for i, part in enumerate(playground_split)})
    return indexes


def temporal_fold_playground_n_tests_split(raw_indexes,
                                           min_n_train_examples=default_pars.validation_min_n_train_examples,
                                           dev_n_sets=default_pars.validation_dev_n_sets,
                                           test_fraction=default_pars.validation_pars_test_fraction,
                                           test_n_sets=default_pars.validation_pars_test_n_sets,
                                           train_name=default_pars.train_key,
                                           dev_name=default_pars.dev_key,
                                           playground_name=default_pars.playground_key,
                                           test_name=default_pars.test_key):
    """Generate indexes that split data into a playground (with temporal folds) and n test sets.

    There is only one playground, which contains train and dev sets, and has no overlap with test sets.
    Playground is split using temporal validation:
    The first 'min_n_train_examples' examples are the first train set.
    The remaining examples in the playground are split homogeneously in 'dev_n_sets' sets (the dev sets).
    The corresponding train set of each of these sets will be made of all previous examples.
    There are hence 'dev_n_sets' non-overlapping dev sets, and the same number of (overlapping) train sets.
    Train sets have different lengths (the first one is the shortest, with 'min_n_train_examples'), and subsequent train
    sets have more and more examples.

    Parameters
    ----------
    raw_indexes : array_like
        All indexes of data. This could simply be the output of 'data.index' (assuming data is a pandas dataframe).
    min_n_train_examples : int
        Minimum number of examples in a train set. It will be the exact number of examples of the first train set.
        All subsequent train sets will be larger than the first one.
    dev_n_sets : int
        Number of temporal folds.
    test_fraction : float
        Fraction of data to use for test sets.
    test_n_sets : int
        Number of test sets.
    train_name : str
        Name given to the train set (usually 'train').
    dev_name : str
        Name given to the dev set (usually 'dev').
    playground_name : str
        Name given to the playground (usually 'playground').
    test_name : str
        Name given to the test set (usually 'test').

    Returns
    -------
    indexes : dict
        Indexes to use for validation. It contains one playground (named 'playground') and test sets (named 'test_0',
        ..., 'test_n'). The indexes of train and dev sets (contained in the playground) are also given (named 'train_0',
        ..., 'train_k' and 'dev_0', ..., 'dev_k', respectively).

    """
    # Split data set into playground and test set(s) without shuffling or stratifying (so they keep their order).
    indexes = train_n_tests_split(raw_indexes=raw_indexes, test_fraction=test_fraction, test_n_sets=test_n_sets,
                                  labels=None, shuffle=False, random_state=None, train_name=playground_name,
                                  test_name=test_name)
    # Split playground into k train and k dev temporal folds.
    playground_split = temporal_folds_split(indexes[playground_name], min_n_train_examples=min_n_train_examples,
                                            dev_n_sets=dev_n_sets)
    indexes.update({f'{train_name}_{i}': part[0] for i, part in enumerate(playground_split)})
    indexes.update({f'{dev_name}_{i}': part[1] for i, part in enumerate(playground_split)})
    return indexes


def validate_indexes(indexes, train_name=default_pars.train_key,
                     dev_name=default_pars.dev_key,
                     playground_name=default_pars.playground_key,
                     test_name=default_pars.test_key):
    # For convenience, collect all indexes in lists.
    train_indexes = []
    dev_indexes = []
    test_indexes = []
    for key in indexes:
        if key.startswith(f'{train_name}_'):
            train_indexes.extend(indexes[key])
        elif key.startswith(f'{dev_name}_'):
            dev_indexes.extend(indexes[key])
        elif key.startswith(f'{test_name}_'):
            test_indexes.extend(indexes[key])

    # Since there can be repetition among indexes in train sets, take unique.
    train_indexes = np.unique(train_indexes)

    # Validations that need to be passed:
    validations = []

    # The set of playground examples coincides with the union of all train and dev sets.
    validations.append(set(indexes[playground_name]) == (set(train_indexes) | set(dev_indexes)))

    # Train indexes and test indexes do not overlap.
    validations.append((set(train_indexes) & set(test_indexes)) == set())

    # For each of the folds in playground, train and dev do not overlap.
    folds = [int(key.split('_')[-1]) for key in indexes if key.startswith(f'{dev_name}_')]
    for fold in folds:
        validations.append(((set(indexes[f'{train_name}_{fold}']) & set(indexes[f'{dev_name}_{fold}'])) == set()))

    # There is no overlap among dev sets.
    validations.append(len(dev_indexes) == len(np.unique(dev_indexes)))

    # There is no overlap among test sets.
    validations.append(len(test_indexes) == len(np.unique(test_indexes)))

    check_validations = np.sum(validations) == len(validations)
    return check_validations
