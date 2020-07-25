from modev import approaches
from modev import default_pars
from modev import etl
from modev import evaluation
from modev import execution
from modev import exploration
from modev import selection
from modev import validation


# TODO: separate approaches into approaches_function and approaches_pars.
load_function = etl.load_local_file
load_pars = {'data_file': default_pars.example_data_path,
             'random_state': default_pars.random_state
             }
validation_function = validation.k_fold_playground_n_tests_split
validation_pars = {'test_fraction': default_pars.validation_pars_test_fraction,
                   'test_n_sets': default_pars.validation_pars_test_n_sets,
                   'playground_n_folds': default_pars.validation_pars_playground_n_folds,
                   'random_state': default_pars.random_state,
                   }
execution_function = execution.execute_model
execution_pars = {'target': 'color',
                  'test_mode': False}
evaluation_function = evaluation.evaluate_predictions
evaluation_pars = {'metrics': ['accuracy'],
                   }
exploration_function = exploration.GridSearch
exploration_pars = {'do_not_expand': None,
                    }
selection_function = selection.model_selection
selection_pars = {'main_metric': 'accuracy',
                  }
approaches = {'dummy_predictor': {'approach_function': approaches.DummyPredictor,
                                  'approach_pars': {'dummy_prediction': ['red', 'blue', 'green'],
                                                    },
                                  },
              'random_predictor': {'approach_function': approaches.RandomChoicePredictor,
                                   'approach_pars': {'random_state': default_pars.random_state,
                                                     },
                                   },
              }
