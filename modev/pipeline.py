"""Main modev module, that contains Pipeline.

"""
import logging

from modev import common
from modev import default_pars
from modev import execution
from modev import plotting
from modev import templates
from modev import validation
from modev.templates import default

app_name_key = default_pars.approach_name_key
function_key = default_pars.function_key

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.WARNING)


def _check_requirements(previous_requirements, error_message):
    if any([requirement is None for requirement in previous_requirements]):
        logging.error(error_message)


def _split_function_and_pars(inputs):
    function = inputs[function_key]
    pars = {par: inputs[par] for par in inputs if par != function_key}
    return function, pars


def _split_approaches_function_and_pars(approaches):
    function = {app[app_name_key]: app[function_key] for app in approaches}
    pars = {app[app_name_key]: {par: app[par] for par in app if par not in [function_key, app_name_key]}
            for app in approaches}
    return function, pars


def _override_default_inputs(given_inputs, default_inputs):
    # If function_key is not given in pars, default function will be used.
    # Therefore, ensure all required parameters are taken from default, except the ones explicitly given in pars.
    if given_inputs is None:
        new_inputs = default_inputs.copy()
    else:
        if function_key in given_inputs:
            new_inputs = given_inputs.copy()
        else:
            new_inputs = default_inputs.copy()
            new_inputs.update(given_inputs)
    return new_inputs


class Pipeline:
    def __init__(self, *,
                 load_inputs=None,
                 validation_inputs=None,
                 execution_inputs=None,
                 evaluation_inputs=None,
                 exploration_inputs=None,
                 selection_inputs=None,
                 approaches_inputs=None,
                 results_file=None,
                 save_every=10):
        """Model development pipeline.

        The arguments accepted by Pipeline refer to the usual ingredients in a data science project (data loading,
        evaluation metrics, model selection method...). For any of those arguments, one can either use a default
        function or provide a custom function.
        * To provide a custom function in any of the dictionaries, use the 'function' key. All parameters required by a
        custom function can be given in the same dictionary.
        * If 'function' is not specified, a default function (taken from one of modev's modules) is used. Arguments of
        the function can be given in the same dictionary (and if not given, default values are assumed).

        Parameters
        ----------
        load_inputs : dict
            Inputs related to data loading.
            See documentation of etl.load_local_file.
        validation_inputs : dict
            Inputs related to validation method (e.g. k-fold or temporal-fold cross-validation).
            See documentation of validation.k_fold_playground_n_tests_split.
        execution_inputs : dict
            Inputs related to the execution of approaches (by default, an approach consists of a class with a 'fit' and
            a 'predict' method).
            See documentation of execution.execute_model.
        evaluation_inputs : dict
            Inputs related to evaluation metrics.
            See documentation of evaluation.evaluate_predictions.
        exploration_inputs : dict
            Inputs related to the method to explore the parameter space (e.g. grid search or random search).
            See documentation of exploration.GridSearch.
            Note: In this argument, 'function' is actually a class.
        selection_inputs : dict
            Inputs related to the model selection method.
            See documentation of selection.model_selection.
        approaches_inputs : list of dicts
            List of approaches to be used. Each element in the list is a dictionary with at least two keys:
            * 'approach_name': Name of the approach (str).
            * 'function': Actual approach (usually, a class with 'fit' and 'predict' methods).
            * Any other given key will be assumed to be arguments of the approach.
            See documentation of default approaches (approaches.DummyPredictor and approaches.RandomChoicePredictor).
        results_file : str or None
            Optional path to local file where to store (temporary or finished) results.
        save_every : int
            Save temporary results to file every save_every consecutive iterations. Only relevant if results_file is
            not None.

        Examples
        --------
        To initialise pipeline with all default inputs:
        >>> pipe = Pipeline()
        To run it:
        >>> pipe.run()
        To get the data:
        >>> pipe.get_data()
        To get the indexes of train/dev/test sets:
        >>> pipe.get_indexes()
        To get the evaluation of each of the executions on each of the folds:
        >>> pipe.get_results()
        To plot evaluation results per fold:
        >>> pipe.plot_results()
        To get the final ranking of best approaches (after combining results of different folds):
        >>> pipe.get_selected_models()

        To initialise pipeline with a template experiment (a dictionary with 'load_inputs', 'validation_inputs', etc.):
        >>> experiment = templates.experiment_01.experiment
        >>> pipe = Pipeline(**experiment)
        And to run it follow the previous example.

        """
        # If input is not given, take it from default.
        # If input is given without specifying a function, take all required parameters from default except those
        # explicitly given.
        load_inputs = _override_default_inputs(load_inputs, default.load_inputs)
        validation_inputs = _override_default_inputs(validation_inputs, default.validation_inputs)
        execution_inputs = _override_default_inputs(execution_inputs, default.execution_inputs)
        evaluation_inputs = _override_default_inputs(evaluation_inputs, default.evaluation_inputs)
        exploration_inputs = _override_default_inputs(exploration_inputs, default.exploration_inputs)
        selection_inputs = _override_default_inputs(selection_inputs, default.selection_inputs)
        if approaches_inputs is None:
            approaches_inputs = default.approaches_inputs
        # Split dictionaries into two: one for functions/classes and another for parameters.
        self.load_function, self.load_pars = _split_function_and_pars(load_inputs)
        self.validation_function, self.validation_pars = _split_function_and_pars(validation_inputs)
        self.execution_function, self.execution_pars = _split_function_and_pars(execution_inputs)
        self.evaluation_function, self.evaluation_pars = _split_function_and_pars(evaluation_inputs)
        self.exploration_function, self.exploration_pars = _split_function_and_pars(exploration_inputs)
        self.selection_function, self.selection_pars = _split_function_and_pars(selection_inputs)
        self.approaches_function, self.approaches_pars = _split_approaches_function_and_pars(approaches_inputs)
        # Initialise other attributes.
        self.data = None
        self.train_indexes = None
        self.test_indexes = None
        self.results = None
        self.ranking = None
        self.results_file = results_file
        self.save_every = save_every

    requirements_error_message = "Methods have to be executed in the following order:" \
                                 "(1) get_data()" \
                                 "(2) get_indexes()" \
                                 "(3) get_results()" \
                                 "(4) get_selected_models()"

    def get_data(self, reload=False):
        _check_requirements([], self.requirements_error_message)
        if self.data is None or reload:
            self.data = self.load_function(**self.load_pars)
        return self.data

    def get_indexes(self, reload=False):
        _check_requirements([self.data], self.requirements_error_message)
        if (self.train_indexes is None and self.test_indexes is None) or reload:
            self.train_indexes, self.test_indexes = self.validation_function(self.data, **self.validation_pars)
        return self.train_indexes, self.test_indexes

    def get_results(self, reload=False):
        _check_requirements([self.data, self.train_indexes, self.test_indexes], self.requirements_error_message)

        if self.results is None or reload:
            self.results = execution.run_experiment(
                self.data, self.train_indexes, self.test_indexes, self.execution_function, self.execution_pars,
                self.evaluation_function, self.evaluation_pars, self.exploration_function, self.approaches_function,
                self.approaches_pars, results_file=self.results_file, save_every=self.save_every, reload=reload)
        return self.results

    def get_selected_models(self, reload=False):
        _check_requirements([self.results], self.requirements_error_message)
        if self.ranking is None or reload:
            self.ranking = self.selection_function(self.results, **self.selection_pars)
        return self.ranking

    def run(self, reload=False):
        self.get_data(reload)

        self.get_indexes(reload)

        self.get_results(reload)

        self.get_selected_models(reload)

        logging.info('Experiment executed successfully')

        return self.ranking

    def plot_results(self, metrics=None):
        _check_requirements([self.results], self.requirements_error_message)
        if metrics is None:
            metrics = common.get_metrics_from_results(self.results)
        for metric in metrics:
            plotting.metric_vs_folds(self.results, metric)
