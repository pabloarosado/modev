"""Functions related to the validation process, e.g. k-fold or temporal-fold cross-validation.

"""
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

from modev import default_pars

dev_key = default_pars.dev_key
playground_key = default_pars.playground_key
test_key = default_pars.test_key
train_key = default_pars.train_key


def k_folds_split(raw_indexes, n_splits, labels=default_pars.validation_pars_labels,
                  shuffle=default_pars.validation_pars_shuffle, random_state=default_pars.random_state,
                  return_original_indexes=default_pars.validation_pars_return_original_indexes):
    """Splits a raw set of indexes into k train and k dev subsets using k-folding.

    There are k (given by 'n_splits') folds. Each of the folds uses the entire raw set of indexes (either for train or
    for dev). The k dev sets do not overlap, and together they cover the entire raw set. For each fold, the train set is
    made by all examples that are not in the dev set. Hence all train sets of different folds do overlap.

    Parameters
    ----------
    raw_indexes : array_like
        Indexes of data (e.g. data.index, assuming data is a pandas dataframe).
    n_splits : int
        Number of folds.
    labels : list or None
        If not None, the k-folding is stratified; if None, labels are ignored.
    shuffle : bool
        True to shuffle indexes before splitting; False to keep original order.
    random_state : int or None
        Random state for shuffling; Ignored if 'shuffle' is False (in which case, 'random_state' can be set to None).
    return_original_indexes : bool
        True to return original indexes (as given by 'raw_indexes'); False to return new integer indexes (that go from 0
        to the number of elements in raw_indexes).

    Returns
    -------
    parts : list
        K different parts (folds). Each part contains a tuple with:
        (array of indexes in train set for this part, array of indexes in dev set for this part)

    """
    raw_indexes_array = np.array(raw_indexes)
    # To avoid warnings, impose random_state None if there is no shuffling.
    if not shuffle:
        random_state = None
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
    """Splits a raw set of indexes into k train and k dev subsets using temporal-folding.

    We assume a simplistic temporal validation: separate the first 'min_n_train_examples' examples for the first train
    set. Then split the remaining examples homogeneously in 'dev_n_sets' sets; they will be the dev sets. The
    corresponding train set of each of these sets will be made of all previous examples.
    Therefore, dev sets do not overlap. But each of the train sets fully contains the previous train set.

    Parameters
    ----------
    raw_indexes : array_like
        Indexes of data (e.g. data.index, assuming data is a pandas dataframe).
    min_n_train_examples : int
        Minimum number of examples in any train set; This will be the number of examples in the first train set. All
        subsequent train sets will be larger than this.
    dev_n_sets : int
        Number of parts (folds).

    Returns
    -------
    parts : list
        K different parts (folds). Each part contains a tuple with:
        (array of indexes in train set for this part, array of indexes in dev set for this part)

    """
    raw_indexes_array = np.array(raw_indexes)
    first_train = raw_indexes_array[0: min_n_train_examples]

    dev_splits = np.array_split(raw_indexes_array[min_n_train_examples:], dev_n_sets)

    parts = [(first_train, dev_splits[0])]
    for fold in range(1, dev_n_sets):
        dev_fold = dev_splits[fold]
        train_fold = raw_indexes_array[raw_indexes_array < dev_fold[0]]
        parts.append((train_fold, dev_fold))
    return parts


def one_set_n_sets_split(data, test_fraction, test_n_sets, first_set_name, second_set_name,
                         labels=default_pars.validation_pars_labels, shuffle=default_pars.validation_pars_shuffle,
                         random_state=default_pars.random_state):
    """Splits a raw set of indexes into one set (e.g. a playground) and n sets (e.g. test sets).

    The raw indexes are split so that a 'test_fraction' is for test sets (as many as 'test_n_sets'). The rest of the raw
    indexes will be for the first set (e.g. the playground).
    Therefore, there is no overlap between the first part (playground) and the second part (test sets), and there is no
    overlap between the different test sets.

    Parameters
    ----------
    data : pd.DataFrame
        Indexed data (e.g. a dataframe whose index can be accessed with data.index).
    test_fraction : float
        Fraction of data to use for test sets.
    test_n_sets : int
        Number of test sets.
    first_set_name : str
        Name to assign to first part (the one that is not a test set), e.g. 'playground'.
    second_set_name : str
        Name to assign to second part (the test sets), e.g. 'test'.
    labels : list or None
        If not None, splits are stratified; if None, labels are ignored.
    shuffle : bool
        True to shuffle original indexes; False to keep order of raw indexes.
    random_state : int or None
        Random state for shuffling; Ignored if 'shuffle' is False (in which case, 'random_state' can be set to None).

    Returns
    -------
    indexes : dict
        Indexes. It contains the first part (e.g. the 'playground_0') and some test sets (e.g. named 'test_0', ...,
        'test_n').

    """
    # To begin with, the raw dataset is train, and there is only one test set (named 'test_0'), which is empty.
    indexes = {f'{first_set_name}_0': np.array(data.index),
               f'{second_set_name}_0': np.array([], dtype=int)}

    # To avoid warnings, impose random_state None if there is no shuffling.
    if not shuffle:
        random_state = None
    # If 'test_fraction' is not zero, take that fraction from train.
    # The new train will be a fraction 1 - 'test_fraction' of the raw dataset.
    # If labels are given, the splitting will be stratified (otherwise random).
    if test_fraction > 0:
        primary_split = train_test_split(indexes[first_set_name + '_0'], test_size=test_fraction, stratify=labels,
                                         random_state=random_state, shuffle=shuffle)
        indexes[f'{first_set_name}_0'] = primary_split[0]
        indexes[f'{second_set_name}_0'] = primary_split[1]

        # If 'test_n_sets' > 1, we split the test set into 'test_n_sets' sets (named 'test_0', 'test_1', etc.) of
        # approximately equal size.
        # Again, if labels are given, the splitting will be stratified (otherwise random).
        # For convenience, use 'k_folds_split' for this task (and then ignore train parts).
        if test_n_sets > 1:
            test_split = k_folds_split(indexes[f'{second_set_name}_0'], test_n_sets, labels=labels, shuffle=shuffle,
                                       random_state=random_state)
            # Disregard the zeroth part (which is meant for training), and keep the non-overlapping part.
            indexes.update({f'{second_set_name}_{i}': fold[1] for i, fold in enumerate(test_split)})

    return indexes


def _split_train_and_test_indexes(indexes, test_mode):
    if test_mode:
        train_indexes = {int(group.split('_')[-1]): indexes[group] for group in indexes
                         if group.startswith(default_pars.playground_key)}
        test_indexes = {int(group.split('_')[-1]): indexes[group] for group in indexes
                        if group.startswith(default_pars.test_key)}
    else:
        train_indexes = {int(group.split('_')[-1]): indexes[group] for group in indexes
                         if group.startswith(default_pars.train_key)}
        test_indexes = {int(group.split('_')[-1]): indexes[group] for group in indexes
                        if group.startswith(default_pars.dev_key)}

    return train_indexes, test_indexes


def k_fold_playground_n_tests_split(data, playground_n_folds=default_pars.validation_pars_playground_n_folds,
                                    test_fraction=default_pars.validation_pars_test_fraction,
                                    test_n_sets=default_pars.validation_pars_test_n_sets,
                                    labels=default_pars.validation_pars_labels,
                                    shuffle=default_pars.validation_pars_shuffle,
                                    random_state=default_pars.random_state,
                                    test_mode=default_pars.validation_pars_test_mode):
    """Generate indexes that split data into a playground (with k folds) and n test sets.

    There is only one playground, which contains train and dev sets, and has no overlap with test sets.
    Playground is split into k folds, namely k non-overlapping dev sets, and k overlapping train sets.
    Each of the folds contains all data in the playground (part of it in train, and the rest in dev); hence train and
    dev sets of the same fold do not overlap.

    Parameters
    ----------
    data : pd.DataFrame
        Indexed data (e.g. a dataframe whose index can be accessed with data.index).
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
    random_state : int or None
        Random state for shuffling; Ignored if 'shuffle' is False (in which case, 'random_state' can be set to None).
    test_mode : bool
        True to return indexes of the test set; False to return indexes of the dev set.

    Returns
    -------
    train_indexes : dict
        Indexes to use for training on the different k folds, e.g. for 10 folds:
        {'train_0': np.array([...]), 'train_1': np.array([...]), ..., 'train_10': np.array([...])}.
    test_indexes : dict
        Indexes to use for evaluating (either dev or test) on the different k folds, e.g. for 10 folds and if test_mode
        is False:
        {'dev_0': np.array([...]), 'dev_1': np.array([...]), ..., 'dev_10': np.array([...])}.

    """
    # To avoid warnings, impose random_state None if there is no shuffling.
    if not shuffle:
        random_state = None
    # Split data set into playground and test set(s).
    indexes = one_set_n_sets_split(data=data, test_fraction=test_fraction, test_n_sets=test_n_sets,
                                   first_set_name=playground_key, second_set_name=test_key,
                                   labels=labels, shuffle=shuffle, random_state=random_state)
    # Split playground into k train and k dev sets.
    playground_split = k_folds_split(indexes[playground_key + '_0'], playground_n_folds, labels=None, shuffle=True,
                                     random_state=random_state)
    indexes.update({f'{train_key}_{i}': part[0] for i, part in enumerate(playground_split)})
    indexes.update({f'{dev_key}_{i}': part[1] for i, part in enumerate(playground_split)})

    assert validate_indexes(indexes)

    train_indexes, test_indexes = _split_train_and_test_indexes(indexes, test_mode)

    return train_indexes, test_indexes


def temporal_fold_playground_n_tests_split(data,
                                           min_n_train_examples=default_pars.validation_min_n_train_examples,
                                           dev_n_sets=default_pars.validation_dev_n_sets,
                                           test_fraction=default_pars.validation_pars_test_fraction,
                                           test_n_sets=default_pars.validation_pars_test_n_sets,
                                           test_mode=default_pars.validation_pars_test_mode):
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
    data : pd.DataFrame
        Indexed data (e.g. a dataframe whose index can be accessed with data.index).
    min_n_train_examples : int
        Minimum number of examples in a train set. It will be the exact number of examples of the first train set.
        All subsequent train sets will be larger than the first one.
    dev_n_sets : int
        Number of temporal folds.
    test_fraction : float
        Fraction of data to use for test sets.
    test_n_sets : int
        Number of test sets.
    test_mode : bool
        True to return indexes of the test set; False to return indexes of the dev set.

    Returns
    -------
    train_indexes : dict
        Indexes to use for training on the different k folds, e.g. for 10 folds:
        {'train_0': np.array([...]), 'train_1': np.array([...]), ..., 'train_10': np.array([...])}.
    test_indexes : dict
        Indexes to use for evaluating (either dev or test) on the different k folds, e.g. for 10 folds and if test_mode
        is False:
        {'dev_0': np.array([...]), 'dev_1': np.array([...]), ..., 'dev_10': np.array([...])}.

    """
    # Split data set into playground and test set(s) without shuffling or stratifying (so they keep their order).
    indexes = one_set_n_sets_split(data=data, test_fraction=test_fraction, test_n_sets=test_n_sets,
                                   first_set_name=playground_key, second_set_name=test_key,
                                   labels=None, shuffle=False, random_state=None)
    # Split playground into k train and k dev temporal folds.
    playground_split = temporal_folds_split(indexes[playground_key + '_0'], min_n_train_examples=min_n_train_examples,
                                            dev_n_sets=dev_n_sets)
    indexes.update({f'{train_key}_{i}': part[0] for i, part in enumerate(playground_split)})
    indexes.update({f'{dev_key}_{i}': part[1] for i, part in enumerate(playground_split)})

    assert validate_indexes(indexes)

    train_indexes, test_indexes = _split_train_and_test_indexes(indexes, test_mode)

    return train_indexes, test_indexes


def validate_indexes(indexes):
    """Check that indexes fulfil some criteria (e.g. that playground and test set do not overlap).

    Parameters
    ----------
    indexes : dict
        Indexes.

    Returns
    -------
    checks : bool
        True if all checks are fulfilled; False otherwise.

    """
    # For convenience, collect all indexes in lists.
    train_indexes = []
    dev_indexes = []
    test_indexes = []
    for key in indexes:
        if key.startswith(f'{train_key}_'):
            train_indexes.extend(indexes[key])
        elif key.startswith(f'{dev_key}_'):
            dev_indexes.extend(indexes[key])
        elif key.startswith(f'{test_key}_'):
            test_indexes.extend(indexes[key])

    # Since there can be repetition among indexes in train sets, take unique.
    train_indexes = np.unique(train_indexes)

    # Validations that need to be passed:
    checks = True

    # The set of playground examples coincides with the union of all train and dev sets.
    checks &= set(indexes[playground_key + '_0']) == (set(train_indexes) | set(dev_indexes))

    # Train indexes and test indexes do not overlap.
    checks &= (set(train_indexes) & set(test_indexes)) == set()

    # For each of the folds in playground, train and dev do not overlap.
    folds = [int(key.split('_')[-1]) for key in indexes if key.startswith(f'{dev_key}_')]
    for fold in folds:
        checks &= ((set(indexes[f'{train_key}_{fold}']) & set(indexes[f'{dev_key}_{fold}'])) == set())

    # There is no overlap among dev sets.
    checks &= len(dev_indexes) == len(np.unique(dev_indexes))

    # There is no overlap among test sets.
    checks &= len(test_indexes) == len(np.unique(test_indexes))

    return checks
