import logging

from modev import execution
from modev import plotting
from modev import validation
from modev.templates import default

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)


def _check_requirements(previous_requirements, error_message):
    if any([requirement is None for requirement in previous_requirements]):
        logging.error(error_message)

# TODO: Create function to check all required inputs.
# TODO: Put all default values in a default_pars folder, and make default experiment a template.
#  Then ensure all default parameters in functions come from default_pars.


class Pipeline:
    def __init__(self, *,
                 load_function=default.load_function,
                 load_pars=default.load_pars,
                 validation_function=default.validation_function,
                 validation_pars=default.validation_pars,
                 execution_function=default.execution_function,
                 execution_pars=default.execution_pars,
                 evaluation_function=default.evaluation_function,
                 evaluation_pars=default.evaluation_pars,
                 exploration_function=default.exploration_function,
                 exploration_pars=default.exploration_pars,
                 selection_function=default.selection_function,
                 selection_pars=default.selection_pars,
                 approaches=default.approaches):
        self.load_function = load_function
        self.load_pars = load_pars
        self.validation_function = validation_function
        self.validation_pars = validation_pars
        self.execution_function = execution_function
        self.execution_pars = execution_pars
        self.evaluation_function = evaluation_function
        self.evaluation_pars = evaluation_pars
        self.exploration_function = exploration_function
        self.exploration_pars = exploration_pars
        self.selection_function = selection_function
        self.selection_pars = selection_pars
        self.approaches = approaches
        # Initialise other attributes.
        self.metrics = self.evaluation_pars['metrics']
        self.main_metric = self.selection_pars['main_metric']
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
                                                    self.exploration_function, self.approaches)
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
