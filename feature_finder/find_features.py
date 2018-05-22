#!/usr/bin/env python

import csv
# import inspect
# from operator import attrgetter

from feature_finder.plugins import plugins


class Model:

    def __init__(self):
        pass

    def select(data):
        """Get the result of fitting the data on selectors."""
        raise NotImplementedError

    def print_results(self):
        """Prints the model's results."""
        raise NotImplementedError


def setup_model(data_csv):
    raise NotImplementedError
