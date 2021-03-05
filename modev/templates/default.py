"""Default inputs for modev pipeline.

"""
from modev import approaches
from modev import default_pars
from modev import etl
from modev import evaluation
from modev import execution
from modev import exploration
from modev import selection
from modev import validation

load_inputs = {default_pars.function_key: etl.load_local_file,
               'data_file': default_pars.example_data_path,
               'random_state': default_pars.random_state
               }
validation_inputs = {default_pars.function_key: validation.k_fold_playground_n_tests_split,
                     'test_fraction': default_pars.validation_pars_test_fraction,
                     'test_n_sets': default_pars.validation_pars_test_n_sets,
                     'playground_n_folds': default_pars.validation_pars_playground_n_folds,
                     'random_state': default_pars.random_state,
                     'test_mode': default_pars.validation_pars_test_mode,
                     }
execution_inputs = {default_pars.function_key: execution.execute_model,
                    'target': 'color'}
evaluation_inputs = {default_pars.function_key: evaluation.evaluate_predictions,
                     'metrics': ['accuracy'],
                     }
exploration_inputs = {default_pars.function_key: exploration.GridSearch,
                      }
selection_inputs = {default_pars.function_key: selection.model_selection,
                    'main_metric': 'accuracy',
                    }
approaches_inputs = [{default_pars.approach_name_key: 'dummy_predictor',
                      default_pars.function_key: approaches.DummyPredictor,
                      'dummy_prediction': ['red', 'blue', 'green'],
                      },
                     {default_pars.approach_name_key: 'random_predictor',
                      default_pars.function_key: approaches.RandomChoicePredictor,
                      'random_state': default_pars.random_state,
                      },
                     ]
