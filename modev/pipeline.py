import logging

from modev import default
from modev import etl
from modev import validation

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)


class Pipeline:
    manual_inputs = ['data_function', 'validation_function', 'evaluation_function']

    def __init__(self, raw_experiment_file, **kwargs):
        self.raw_experiment_file = raw_experiment_file
        self.kwargs = kwargs
        # Load default experiment and update it with given inputs.
        self.raw_experiment = self._parse_inputs()
        # Initialise other attributes.
        self.data = None
        self.indexes = None

    def _parse_inputs(self):
        # Load default experiment and overwrite inputs given in raw experiment file.
        raw_experiment_inputs = etl.load_raw_experiment(self.raw_experiment_file)
        raw_experiment = default.default_experiment.copy()
        for field in list(raw_experiment):
            if field in raw_experiment_inputs:
                raw_experiment[field] = raw_experiment_inputs[field]
        # Overwrite inputs given manually.
        for kwarg in self.kwargs:
            if kwarg in self.manual_inputs:
                raw_experiment[kwarg] = self.kwargs[kwarg]
            else:
                logging.warning("Unknown keyword argument: %s", kwarg)
        return raw_experiment

    def get_data(self):
        self.data = self.raw_experiment['data_function'](**self.raw_experiment['data_pars'])
        return self.data

    def get_indexes(self):
        self.indexes = self.raw_experiment['validation_function'](self.data.index,
                                                                  **self.raw_experiment['validation_pars'])
        if not validation.validate_indexes(self.indexes):
            logging.warning("Indexes do not pass validations!")
        return self.indexes
