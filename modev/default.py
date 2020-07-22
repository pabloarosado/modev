from modev import approaches
from modev import etl
from modev import evaluation
from modev import execution
from modev import exploration
from modev import selection
from modev import validation

random_state = 1000
test_fraction = 0.2
test_n_sets = 1
playground_n_folds = 4

experiment = {'load_function': etl.load_local_file,
              'load_pars': {'data_file': 'some_example_file',  # TODO: include example file.
                            'random_state': random_state,
                            },
              'validation_function': validation.k_fold_playground_n_tests_split,
              'validation_pars': {'test_fraction': test_fraction,
                                  'test_n_sets': test_n_sets,
                                  'playground_n_folds': playground_n_folds,
                                  'random_state': random_state,
                                  },
              'execution_function': execution.execute_model,
              'execution_pars': {'target': 'example_target_column',
                                 'test_mode': False},
              'evaluation_function': evaluation.evaluate_predictions,
              'evaluation_pars': {'metrics': ['accuracy'],
                                  },
              'exploration_function': exploration.GridSearch,
              'exploration_pars': {'do_not_expand': None,
                                   },
              'selection_function': selection.model_selection,
              'selection_pars': {'main_metric': 'accuracy',
                                 },
              'approaches': {'dummy_predictor': {'approach_function': approaches.DummyPredictor,
                                                 'approach_pars': {'dummy_prediction': 0,
                                                                   },
                                                 },
                             },
              }
