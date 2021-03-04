"""Simple approaches to use for testing purposes, or as baselines.

"""
import numpy as np

from modev import default_pars


class DummyPredictor:
    def __init__(self, dummy_prediction):
        """Predict always the same (a fixed prediction).

        Parameters
        ----------
        dummy_prediction : str or int or float
            Prediction that is returned every time.

        Methods
        -------
        fit
            Does nothing.
        predict
            Repeats the dummy prediction as many times as elements in the given 'test_x'.

        """
        self.dummy_prediction = dummy_prediction

    def fit(self, train_x, train_y):
        """Does nothing.

        Parameters
        ----------
        train_x : pd.DataFrame
            Predictor values of the train set. Ignored for this approach.
        train_y : np.array
            Target values of the train set. Ignored for this approach.

        Returns
        -------
        None

        """
        pass

    def predict(self, test_x):
        """Predict on test set, given the predictor values 'test_x'.

        Parameters
        ----------
        test_x : pd.DataFrame
            Predictor values of the test set.

        Returns
        -------
        prediction : np.array
            Predicted target values of the test set.

        """
        predictions = np.repeat(self.dummy_prediction, len(test_x))
        return predictions


class RandomChoicePredictor:
    def __init__(self, random_state=default_pars.random_state):
        """Predict a random value of the target column from the train set.

        Parameters
        ----------
        random_state : int
            Random state to use when picking elements from the train set.

        Methods
        -------
        fit
            Does nothing with 'train_x' but keeps 'train_y' (to later use it as source of random predictions).
        predict
            Returns values randomly picked from the target of the train set.

        """
        self.random_state = random_state
        self.possible_choices = None

    def fit(self, train_x, train_y):
        """Fit model to train set, given the predictor values 'train_x' and target values 'train_y'.

        Parameters
        ----------
        train_x : pd.DataFrame
            Predictor values of the train set. Ignored for this approach.
        train_y : np.array
            Target values of the train set. They will be used as possible choices for randomly picking predictions.

        Returns
        -------
        None

        """
        _ = train_x
        self.possible_choices = train_y

    def predict(self, test_x):
        """Predict on test set, given the predictor values 'test_x'.

        Parameters
        ----------
        test_x : pd.DataFrame
            Predictor values of the test set.

        Returns
        -------
        prediction : np.array
            Predicted target values of the test set.

        """
        np.random.seed(self.random_state)
        prediction = np.random.choice(self.possible_choices, len(test_x))
        return prediction
