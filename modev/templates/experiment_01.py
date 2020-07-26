"""Some variations with respect to default experiment.

"""

experiment = {'validation_inputs': {'test_fraction': 0.5,
                                    'test_n_sets': 6,
                                    'random_state': 42,
                                    },
              'execution_inputs': {'test_mode': True},
              'evaluation_inputs': {'metrics': ['precision', 'recall', 'f1'],
                                    'average': 'micro',
                                    },
              'selection_inputs': {'main_metric': 'f1',
                                   },
              }
