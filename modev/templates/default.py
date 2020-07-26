from modev import approaches
from modev import default_pars
from modev import etl
from modev import evaluation
from modev import execution
from modev import exploration
from modev import selection
from modev import validation

# TODO: define all keys in default_pars (named with the corresponding group, e.g. validation_function_key='function').
load_inputs = {'function': etl.load_local_file,
               'data_file': default_pars.example_data_path,
               'random_state': default_pars.random_state
               }
validation_inputs = {'function': validation.k_fold_playground_n_tests_split,
                     'test_fraction': default_pars.validation_pars_test_fraction,
                     'test_n_sets': default_pars.validation_pars_test_n_sets,
                     'playground_n_folds': default_pars.validation_pars_playground_n_folds,
                     'random_state': default_pars.random_state,
                     }
execution_inputs = {'function': execution.execute_model,
                    'target': 'color',
                    'test_mode': False}
evaluation_inputs = {'function': evaluation.evaluate_predictions,
                     'metrics': ['accuracy'],
                     }
exploration_inputs = {'function': exploration.GridSearch,
                      'do_not_expand': None,
                      }
selection_inputs = {'function': selection.model_selection,
                    'main_metric': 'accuracy',
                    }
approaches_inputs = [{'approach_name': 'dummy_predictor',
                      'function': approaches.DummyPredictor,
                      'dummy_prediction': ['red', 'blue', 'green'],
                      },
                     {'approach_name': 'random_predictor',
                      'function': approaches.RandomChoicePredictor,
                      'random_state': default_pars.random_state,
                      },
                     ]
