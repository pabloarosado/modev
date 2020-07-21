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
    # Predict target values by randomly picking them from train set.
    def __init__(self, random_state=1000):
        self.random_state = random_state
        self.possible_choices = None

    def fit(self, train_x, train_y):
        self.possible_choices = train_y

    def predict(self, test_x):
        prediction = np.random.choice(self.possible_choices, len(test_x))
        return prediction
