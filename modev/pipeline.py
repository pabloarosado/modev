import logging

from modev import default
from modev import etl
from modev import execution
from modev import utils
from modev import validation

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)


def _override_default_experiment(experiment_module, default_experiment_module):
    # Load default experiment and raw (incomplete) experiment.
    default_experiment = etl.load_experiment(default_experiment_module)
    raw_experiment = etl.load_experiment(experiment_module)
    # Take everything from default experiment and overwrite inputs given in raw experiment.
    experiment = default_experiment.copy()
    for field in list(experiment):
        if field in raw_experiment:
            experiment[field] = raw_experiment[field]
    return experiment


def _check_requirements(previous_requirements, error_message):
    if any([requirement is None for requirement in previous_requirements]):
        logging.error(error_message)


class Pipeline:
    def __init__(self, experiment_file):
        self.experiment_file = experiment_file
        # Initialise other attributes.
        self.experiment_module = None
        self.experiment_description = None
        self.experiment = None
        self.metrics = None
        self.data = None
        self.indexes = None
        self.results = None
        self.ranking = None

    requirements_error_message = "Methods have to be executed in the following order:" \
                                 "(1) get_experiment()" \
                                 "(2) get_data()" \
                                 "(3) get_indexes()" \
                                 "(4) get_results()"

    def get_experiment(self, reload=False):
        _check_requirements([], self.requirements_error_message)
        if self.experiment is None or reload:
            self.experiment_module = etl.load_experiment_module(self.experiment_file)
            self.experiment_description = utils.get_text_from_docstring(self.experiment_module)
            self.experiment = _override_default_experiment(self.experiment_module, default)
            # TODO: maybe there should be a method 'get_metrics', that ensures field 'metrics' exists.
            #  More generally, there should be a function that checks if experiment is well structured.
            self.metrics = self.experiment['evaluation_pars']['metrics']
        return self.experiment

    def get_data(self, reload=False):
        _check_requirements([self.experiment], self.requirements_error_message)
        if self.data is None or reload:
            self.data = self.experiment['load_function'](**self.experiment['load_pars'])
        return self.data

    def get_indexes(self, reload=False):
        _check_requirements([self.experiment, self.data], self.requirements_error_message)
        if self.indexes is None or reload:
            self.indexes = self.experiment['validation_function'](self.data.index,
                                                                  **self.experiment['validation_pars'])
            if not validation.validate_indexes(self.indexes):
                logging.warning("Indexes do not pass validations!")
        return self.indexes

    def get_results(self, reload=False):
        _check_requirements([self.experiment, self.data], self.requirements_error_message)
        if self.results is None or reload:
            self.results = execution.run_experiment(self.experiment, self.data, self.indexes,
                                                    self.experiment['execution_function'])
        return self.results

    def get_selected_models(self, reload=False):
        _check_requirements([self.experiment, self.data, self.results], self.requirements_error_message)
        if self.ranking is None or reload:
            self.ranking = self.experiment['selection_function'](self.results, self.metrics,
                                                                 **self.experiment['selection_pars'])
        return self.ranking

    def run(self, reload=False):
        self.get_experiment(reload)

        self.get_data(reload)

        self.get_indexes(reload)

        self.get_results(reload)

        self.get_selected_models(reload)

        logging.info('Experiment executed successfully')

        return self.ranking
