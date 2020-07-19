import logging

from modev import default
from modev import etl
from modev import validation

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)


class Pipeline:
    def __init__(self, raw_experiment_file):
        self.raw_experiment_file = raw_experiment_file
        # Initialise other attributes.
        self.raw_experiment = None
        self.data = None
        self.indexes = None

    def get_raw_experiment(self):
        # Load default experiment and overwrite inputs given in raw experiment file.
        raw_experiment_inputs = etl.load_raw_experiment(self.raw_experiment_file)
        raw_experiment = default.default_experiment.copy()
        for field in list(raw_experiment):
            if field in raw_experiment_inputs:
                raw_experiment[field] = raw_experiment_inputs[field]
        self.raw_experiment = raw_experiment
        return self.raw_experiment

    def get_data(self):
        self.data = self.raw_experiment['load_function'](**self.raw_experiment['load_pars'])
        return self.data

    def get_indexes(self):
        self.indexes = self.raw_experiment['validation_function'](self.data.index,
                                                                  **self.raw_experiment['validation_pars'])
        if not validation.validate_indexes(self.indexes):
            logging.warning("Indexes do not pass validations!")
        return self.indexes

    def run(self):
        if self.raw_experiment is None:
            self.get_raw_experiment()

        if self.data is None:
            self.get_data()

        if self.indexes is None:
            self.get_indexes()
