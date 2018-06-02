"""
Linear regression model selects most predictive features.

Existing test files assume that
- the last column of the CSV is the y-column
- the columns have no headers and are ordered by error or accuracy
  (index 0 being the least predictive, n being highest).
"""

import os
from csv import reader
from unittest import TestCase

from feature_finder.find_features import Model, setup_data


class TestFindFeatures(TestCase):

    @classmethod
    def _get_test_files(cls, model_type, plugins):
        files = os.listdir(cls.dir_path + model_type)
        if plugins:
            return [cls.dir_path + model_type + '/' + files[i] for i in range(len(files))
                    if 'plugins' in files[i]]
        else:
            return [cls.dir_path + model_type + '/' + files[i] for i in range(len(files))
                    if 'plugins' not in files[i]]

    @classmethod
    def setUpClass(cls):
        cls.dir_path = os.path.dirname(os.path.realpath(__file__)) + '/sample_data/'
        cls.linear_test_files = cls._get_test_files('linear', False)
        cls.linear_test_files_plugins = cls._get_test_files('linear', True)
        cls.logistic_test_files = cls._get_test_files('logistic', False)
        cls.logistic_test_files_plugins = cls._get_test_files('logistic', True)
        cls.test_plugins = ['string_length']


    def _test_model(self, model_type, test_files, plugins, error=True):
        for f in test_files:
            header = True if 'header' in f else False
            with open(f) as fp:
                n_cols = len(next(reader(fp)))

            data, y = setup_data(f, header, n_cols - 1)
            model = Model(model_type, plugins)
            results = model.select(data, y)

            if error:
                # We are using error to determine the predictive value of columns,
                # so the higher values for the results are less predictive.
                ordered_results = [results[i] for i in range(len(results))]
            else:
                # We are using accuracy to determine the predictive value of columns,
                # so the higher values for the results are more predictive.
                ordered_results = [results[len(results) - i - 1] for i in range(len(results))]
            assert all(lower >= higher for lower, higher in
                       zip(ordered_results, ordered_results[1:]))

    def test_linear_regression(self):
        """Linear regression model selects most predictive features."""
        self._test_model('linear', self.linear_test_files, plugins=None)

    def test_linear_regression_plugins(self):
        """Linear regression model selects most predictive features."""
        self._test_model('linear', self.linear_test_files_plugins, plugins=self.test_plugins)

    def test_logistic_regression(self):
        """Logistic regression model selects most predictive features."""
        self._test_model('logistic', self.logistic_test_files, plugins=None, error=False)

    def test_logistic_regression_plugins(self):
        """Logistic regression model selects most predictive features."""
        self._test_model('logistic', self.logistic_test_files_plugins, plugins=self.test_plugins, error=False)
