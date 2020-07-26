import logging

from modev import default_pars
from modev import execution
from modev import plotting
from modev import validation
from modev.templates import default

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)


def _check_requirements(previous_requirements, error_message):
    if any([requirement is None for requirement in previous_requirements]):
        logging.error(error_message)

# TODO: Create function to check all required inputs.


def _split_function_and_pars(inputs, function_key=default_pars.function_key):
    function = inputs[function_key]
    pars = {par: inputs[par] for par in inputs if par != function_key}
    return function, pars


def _split_approaches_function_and_pars(approaches, function_key=default_pars.function_key,
                                        app_name_key=default_pars.approach_name_key):
    function = {app[app_name_key]: app[function_key] for app in approaches}
    pars = {app[app_name_key]: {par: app[par] for par in app if par not in [function_key, app_name_key]}
            for app in approaches}
    return function, pars


def _override_default_inputs(given_inputs, default_inputs, function_key=default_pars.function_key):
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
                 approaches_inputs=None):
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
        self.metrics = self.evaluation_pars[default_pars.metrics_key]
        self.main_metric = self.selection_pars[default_pars.main_metric_key]
        self.data = None
        self.indexes = None
        self.results = None
        self.ranking = None

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
        if self.indexes is None or reload:
            self.indexes = self.validation_function(self.data.index, **self.validation_pars)
            if not validation.validate_indexes(self.indexes):
                logging.warning("Indexes do not pass validations!")
        return self.indexes

    def get_results(self, reload=False):
        _check_requirements([self.data, self.indexes], self.requirements_error_message)
        if self.results is None or reload:
            self.results = execution.run_experiment(self.data, self.indexes, self.validation_pars,
                                                    self.execution_function, self.execution_pars,
                                                    self.evaluation_function, self.evaluation_pars,
                                                    self.exploration_function, self.approaches_function,
                                                    self.approaches_pars)
        return self.results

    def get_selected_models(self, reload=False):
        _check_requirements([self.data, self.results], self.requirements_error_message)
        if self.ranking is None or reload:
            self.ranking = self.selection_function(self.results, self.metrics, **self.selection_pars)
        return self.ranking

    def run(self, reload=False):
        self.get_data(reload)

        self.get_indexes(reload)

        self.get_results(reload)

        self.get_selected_models(reload)

        logging.info('Experiment executed successfully')

        return self.ranking

    def plot_results(self):
        plotting.metric_vs_folds(self.results, self.main_metric)
