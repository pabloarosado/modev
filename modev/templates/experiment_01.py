"""Some variations with respect to default experiment.

"""

experiment = {'validation_pars': {'test_fraction': 0.5,
                                  'test_n_sets': 6,
                                  'playground_n_folds': 4,
                                  'random_state': 42,
                                  },
              'execution_pars': {'target': 'color',
                                 'test_mode': True},
              'evaluation_pars': {'metrics': ['precision', 'recall', 'f1'],
                                  'average': 'micro',
                                  },
              'exploration_pars': {'do_not_expand': None,
                                   },
              'selection_pars': {'main_metric': 'f1',
                                 },
              }
