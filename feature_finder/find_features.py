#!/usr/bin/env python

import csv
import inspect

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from feature_finder.plugins import plugins

MODELS = {
    'linear': {'model': LinearRegression, 'error': root_mean_squared_error},
    'logistic': {'model': LogisticRegression, 'error': accuracy},
}


class Model:
    test_size = 0.2
    k_fold = 5

    def __init__(self, model_type, plugins=None):
        if model_type not in MODELS:
            raise TypeError(
                "Please specify one of the following model types: {}."
                .format(', '.join(MODELS.keys())))
        self.model = MODELS[model_type]['model']()
        self.plugins = plugins or []

    def select(self, data, y_column):
        """Get the result of fitting the data on selectors."""
        if self.plugins:
            data = self.customize(data)

        train, test = train_test_split(data, test_size=self.test_size)
        x = data.drop(y_column)
        y = data[y_column]
        raise NotImplementedError

    def fit_model(self, features, y):
        self.model.fit(features, y)

    def customize(self, data):
        """Apply plugins."""
        plugins_functions = [
            obj for name, obj in inspect.getmembers(plugins)
            if inspect.isfunction(obj)
        ]
        for fn in plugins_functions:
            try:
                data = fn(data)
            except Exception:
                raise plugins.PluginException(
                    'Error with plugin "{}".'.format(fn.__name__))
        return data

    def print_results(self):
        """Prints the model's results."""
        raise NotImplementedError


def validate(data, header, y):
    if header:
        assert y in data.columns, 'Requested y-column not found in CSV ({})'.format(y)
    else:
        try:
            y = int(y)
            assert y in data.columns, 'Requested y index position not found in CSV ({})'.format(y)
        except (TypeError, ValueError, KeyError) as exc:
            print('Error parsing index position of CSV ({})'.format(y))
            raise exc


def setup_data(data_csv, header, y):
    try:
        data = pd.read_csv(data_csv) if header else pd.read_csv(data_csv, header=None)
        y = int(y) if header else int(y)
    except Exception as exc:
        print('Error reading CSV: {}.'.format(exc))
        exit(1)
    else:
        validate(data, header, y)
        return data, y
