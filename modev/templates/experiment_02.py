"""Template experiment using temporal-fold cross validation instead of k-fold cross-validation.

"""
from modev.validation import temporal_fold_playground_n_tests_split

experiment = {'validation_inputs': {'function': temporal_fold_playground_n_tests_split,
                                    'min_n_train_examples': 20,
                                    'dev_n_sets': 4,
                                    'test_fraction': 0.5,
                                    'test_n_sets': 2,
                                    'test_mode': False,
                                    },
              'evaluation_inputs': {'metrics': ['precision', 'recall', 'f1'],
                                    'average': 'micro',
                                    },
              'selection_inputs': {'main_metric': 'f1',
                                   },
              }
