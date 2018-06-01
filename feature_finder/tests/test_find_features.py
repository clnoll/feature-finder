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
    def _get_test_files(cls, model_type):
        files = os.listdir(cls.dir_path + model_type)
        return [cls.dir_path + model_type + '/' + files[i] for i in range(len(files))]

    @classmethod
    def setUpClass(cls):
        cls.dir_path = os.path.dirname(os.path.realpath(__file__)) + '/sample_data/'
        cls.linear_test_files = cls._get_test_files('linear')
        cls.logistic_test_files = cls._get_test_files('logistic')


    def _test_model(self, model_type, test_files, error=True):
        for f in test_files:
            header = True if 'header' in f else False
            with open(f) as fp:
                n_cols = len(next(reader(fp)))

            data, y = setup_data(f, header, n_cols - 1)
            model = Model(model_type)
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
        self._test_model('linear', self.linear_test_files)

    def test_logistic_regression(self):
        """Logistic regression model selects most predictive features."""
        self._test_model('logistic', self.logistic_test_files, error=False)
