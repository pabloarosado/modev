"""Template experiment which contains some variations with respect to the default inputs.

"""

experiment = {'validation_inputs': {'test_fraction': 0.5,
                                    'test_n_sets': 6,
                                    'random_state': 42,
                                    'test_mode': True,
                                    },
              'evaluation_inputs': {'metrics': ['precision', 'recall', 'f1'],
                                    'average': 'micro',
                                    },
              'selection_inputs': {'main_metric': 'f1',
                                   },
              }
