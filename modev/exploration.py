"""Functions related to the exploration of the parameter space.

"""
import itertools
import numpy as np
import pandas as pd

from modev import default_pars


def expand_parameter_grid(grid, fixed_pars=default_pars.exploration_pars_fixed_pars):
    pars_names = list(grid)
    pars_lists = []
    if fixed_pars is None:
        fixed_pars = []
    for key in pars_names:
        value = grid[key]
        if (type(value) == list) & (key not in fixed_pars):
            pars_lists.append(value)
        else:
            pars_lists.append([value])
    expanded_lists = list(itertools.product(*pars_lists))
    pars = []
    for combination in expanded_lists:
        odict = {pars_names[i]: combination[i] for i in range(len(pars_names))}
        pars.append(odict)
    return pars


def _split_approaches_name_and_pars(approaches_pars, fixed_pars_key=default_pars.fixed_pars_key):
    app_names = []
    app_pars = []
    for name in approaches_pars:
        pars = approaches_pars[name]
        fixed_pars = None
        if fixed_pars_key in pars:
            fixed_pars = pars[fixed_pars_key]
        pars = {par: pars[par] for par in pars if par != fixed_pars_key}
        expanded_grid = expand_parameter_grid(pars, fixed_pars=fixed_pars)
        app_pars.extend(expanded_grid)
        app_names.extend([name] * len(expanded_grid))
    return app_names, app_pars


class GridSearch:
    def __init__(self, approaches_pars, folds, metrics):
        """Grid search exploration of the parameter space.

        Parameters
        ----------
        approaches_pars : dict
            Dictionaries of approaches. Each key corresponds to one approach name, and the value is a dictionary.
            This inner dictionary of an individual approach has one key per parameter, and the value is a list of
            parameter values to explore.
        folds : list of ints
            List of folds (e.g. [0, 1, 2, 3]).
        metrics : list
            List of metrics.

        """
        self.approaches_pars = approaches_pars
        self.folds = folds
        self.metrics = metrics
        self.pars_folds = None
        self.selection_to_execute = None
        self.next_point_generator = None

    def initialise_results(self):
        # TODO: Generalise this to save to/load from file.
        app_names, app_pars = _split_approaches_name_and_pars(self.approaches_pars)
        app_ids = np.arange(len(app_pars))
        # Repeat each pars combination for each fold.
        pars_folds = pd.DataFrame(np.repeat(app_pars, len(self.folds)), columns=[default_pars.pars_key])
        pars_folds[default_pars.approach_key] = np.repeat(app_names, len(self.folds))
        pars_folds[default_pars.id_key] = np.repeat(app_ids, len(self.folds))
        pars_folds[default_pars.fold_key] = np.tile(self.folds, len(app_pars))
        # Add a column for each of the evaluation metrics.
        self.pars_folds = pars_folds
        # Add metrics to results dataframe.
        for metric in self.metrics:
            pars_folds[metric] = np.nan
        return self.pars_folds

    def select_executions_left(self):
        # TODO: Select not executed: unexplored_sel = self.pars_folds['executed'].isnull()
        self.selection_to_execute = np.ones(len(self.pars_folds), dtype=bool)
        n_iterations = len(self.selection_to_execute[self.selection_to_execute])
        return n_iterations

    def _next_point_finder(self):
        for i, row in self.pars_folds[self.selection_to_execute].iterrows():
            yield i, row

    def get_next_point(self):
        if self.next_point_generator is None:
            self.next_point_generator = self._next_point_finder()
        return next(self.next_point_generator)

# TODO: Create RandomSearch with similar structure.
# TODO: Create AutoSearch with similar structure.
#  In this case, _next_point_finder can use self.pars_folds at any time to see explored points and decide next.
