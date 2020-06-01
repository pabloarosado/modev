from modev import etl
from modev import exploration
from modev import validation
from modev import evaluation

default_experiment = {'data_function': etl.load_local_file,
                      'data_pars': {'data_file': None, 'random_state': 1000},
                      'validation_function': validation.folding,
                      'validation_pars': {'type': 'k_fold', 'n_folds': 4, 'random_state': 1000},
                      'evaluation_function': evaluation.evaluate_predictions,
                      'evaluation_pars': {'metrics': ['precision', 'accuracy']},
                      'exploration_function': exploration.grid_search,
                      'exploration_pars': {'do_not_expand': None},
                      'approaches': [{}],
                      }
