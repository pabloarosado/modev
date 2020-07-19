import numpy as np


class DummyPredictor:
    # Always predict the same thing (given as input).
    def __init__(self, dummy_prediction):
        self.dummy_prediction = dummy_prediction

    def fit(self, train_x, train_y):
        pass

    def predict(self, test_x):
        predictions = np.repeat(self.dummy_prediction, len(test_x))
        return predictions


class RandomChoicePredictor:
    # TODO: take dummy par and predict by picking random point of train set.
    def __init__(self):
        pass

    def fit(self, train_x, train_y):
        pass

    def predict(self, test_x):
        pass
