import os
from unittest import TestCase


class TestFindFeatures(TestCase):

    def setUp(self):
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.sample_data_csv = None

    def test_linear_regression(self):
        """Linear regression model selects most predictive features."""
       raise NotImplementedError

   def test_logistic_regression(self):
        """Logistic regression model selects most predictive features."""
       raise NotImplementedError
