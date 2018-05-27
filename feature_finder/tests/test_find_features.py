import os
from unittest import TestCase

from feature_finder.find_features import Model, setup_data


class TestFindFeatures(TestCase):

    def setUp(self):
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.sample_data_csv = None

    def test_validate(self):
        """`validate_data` raises exceptions on errors in input format."""
        raise NotImplementedError

    def test_setup_data(self):
        """`setup_data` returns a dataframe and y-column name in the dataframe."""
        raise NotImplementedError

    def test_linear_regression(self):
        """Linear regression model selects most predictive features."""
        raise NotImplementedError

    def test_logistic_regression(self):
        """Logistic regression model selects most predictive features."""
        raise NotImplementedError
