from modev import approaches
from modev import etl
from modev import evaluation
from modev import exploration
from modev import validation

default_random_state = 1000
default_test_fraction = 0.2
default_test_n_sets = 1
default_playground_n_folds = 4

default_experiment = {'load_function': etl.load_local_file,
                      'load_pars': {'data_file': None,
                                    'random_state': default_random_state,
                                    },
                      'validation_function': validation.k_fold_playground_n_tests_split,
                      'validation_pars': {'test_fraction': default_test_fraction,
                                          'test_n_sets': default_test_n_sets,
                                          'playground_n_folds': default_playground_n_folds,
                                          'random_state': default_random_state,
                                          },
                      'evaluation_function': evaluation.evaluate_predictions,
                      'evaluation_pars': {'metrics': ['accuracy'],
                                          },
                      'exploration_function': exploration.grid_search,
                      'exploration_pars': {'do_not_expand': None,
                                           },
                      'approaches': {'dummy_predictor': {'approach_function': approaches.DummyPredictor,
                                                         'approach_pars': {'dummy_prediction': 0,
                                                                           },
                                                         },
                                     },
                      }
