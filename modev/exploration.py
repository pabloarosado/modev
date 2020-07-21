import itertools
import numpy as np
import pandas as pd


def expand_parameter_grid(grid, fixed_pars=None):
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


def expand_approaches_grids(approaches):
    approaches_names = []
    approaches_functions = []
    approaches_pars = []
    for app_name in approaches:
        app_pars = approaches[app_name]['approach_pars']
        app_function = approaches[app_name]['approach_function']
        # app_pars['approach_name'] = app_name
        fixed_pars = None
        if 'fixed_pars' in app_pars:
            fixed_pars = app_pars['fixed_pars']
        pars = {par: app_pars[par] for par in app_pars if par != 'fixed_pars'}
        expanded_grid = expand_parameter_grid(pars, fixed_pars=fixed_pars)
        approaches_pars.extend(expanded_grid)
        approaches_names.extend([app_name] * len(expanded_grid))
        approaches_functions.extend([app_function] * len(expanded_grid))
    return approaches_names, approaches_functions, approaches_pars


class GridSearch:
    def __init__(self, approaches_grid, folds, metrics):
        self.approaches_grid = approaches_grid
        self.folds = folds
        self.metrics = metrics
        self.pars_folds = None
        self.selection_to_execute = None
        self.next_point_generator = None

    def initialise_results(self):
        # TODO: Generalise this to save to/load from file.
        app_names, app_functions, app_pars = expand_approaches_grids(self.approaches_grid)
        app_ids = np.arange(len(app_pars))
        # Repeat each pars combination for each fold.
        pars_folds = pd.DataFrame(np.repeat(app_pars, len(self.folds)), columns=['pars'])
        pars_folds['approach'] = np.repeat(app_names, len(self.folds))
        pars_folds['id'] = np.repeat(app_ids, len(self.folds))
        pars_folds['fold'] = np.tile(self.folds, len(app_pars))
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
