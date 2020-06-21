from modev import etl
from modev import exploration
from modev import validation
from modev import evaluation

default_random_state = 1000
default_test_fraction = 0.2
default_n_folds = 4

default_experiment = {'data_function': etl.load_local_file,
                      'data_pars': {'data_file': None, 'random_state': default_random_state},
                      'validation_function': validation.folding,
                      'validation_pars': {'type': 'k_fold', 'test_fraction': default_test_fraction,
                                          'n_folds': default_n_folds,
                                          'random_state': default_random_state},
                      'evaluation_function': evaluation.evaluate_predictions,
                      'evaluation_pars': {'metrics': ['precision', 'accuracy']},
                      'exploration_function': exploration.grid_search,
                      'exploration_pars': {'do_not_expand': None},
                      'approaches': [{}],
                      }
