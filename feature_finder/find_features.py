#!/usr/bin/env python

import inspect

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from feature_finder.plugins import plugins
from feature_finder.utils import (root_mean_squared_error,
                                  accuracy,
                                  precision,
                                  recall,
                                  false_alarm)


class BaseModel:
    test_size = 0.2
    k_fold = 5

    def __init__(self, plugins=None):
        self.model = self.model_cls()
        self.plugins = plugins or []

    def select(self, data, y_column):
        """Get the result of fitting the data on selectors."""
        if self.plugins:
            data = self.customize(data)

        train, test = train_test_split(data, test_size=self.test_size)
        x_train = train.drop(y_column, axis=1)
        y_train = train[y_column]
        x_test = test.drop(y_column, axis=1)
        y_test = test[y_column]

        return {c: self.get_feature_error(c, x_train, y_train, x_test, y_test)
                for c in x_train.columns}

    def get_feature_error(self, c, x_train, y_train, x_test, y_test):
        self.model.fit(np.array(x_train[c]).reshape(-1, 1), y_train)
        predicted = self.model.predict(np.array(x_test[c]).reshape(-1, 1))
        return self.__class__.error(y_test, predicted)

    def customize(self, data):
        """Apply plugins."""
        plugins_functions = [
            obj for name, obj in inspect.getmembers(plugins)
            if inspect.isfunction(obj) and name in self.plugins
        ]
        for fn in plugins_functions:
            try:
                data = fn(data)
            except Exception:
                raise plugins.PluginException(
                    'Error with plugin "{}".'.format(fn.__name__))
        return data


class LinearRegressionModel(BaseModel):
    model_cls = LinearRegression
    error = root_mean_squared_error
    best_selector = 0


class LogisticRegressionModel(BaseModel):
    model_cls = LogisticRegression
    error = accuracy
    best_selector = -1


MODELS = {'linear': LinearRegressionModel, 'logistic': LogisticRegressionModel}


def get_model(name, plugins):
    if name not in MODELS:
        raise ValueError(
            "Please specify one of the following model types: {}."
            .format(', '.join(MODELS.keys())))
    return MODELS[name](plugins)


def validate(data, header, y):
    if header:
        assert y in data.columns, 'Requested y-column not found in CSV ({})'.format(y)
    else:
        try:
            y = int(y)
            assert y in data.columns, 'Requested y index position not found in CSV ({})'.format(y)
        except (TypeError, ValueError) as exc:
            print('Error parsing index position of CSV ({})'.format(y))
            raise exc


def setup_data(data_csv, header, y):
    try:
        data = pd.read_csv(data_csv, header=(header or None))
        y = y if header else int(y)
    except Exception as exc:
        print('Error reading CSV: {}.'.format(exc))
        exit(1)
    else:
        validate(data, header, y)
        return data, y


def print_stats(results, model_type):
    """Prints the error for the model's features."""
    features = sorted(results.values())
    row_format = '{:<15} {:^15}'
    formatted_result_type = MODELS[model_type].error.__name__.replace('_', ' ').title()
    print(row_format.format('Column', formatted_result_type))
    for column, error in results.items():
        print(row_format.format(
            str(column) + '*' if results[column] == features[MODELS[model_type].best_selector]
            else str(column),
            round(error, 2)))
