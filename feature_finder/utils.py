import numpy as np


# Utilities for linear regression error

def root_mean_squared_error(t, p):
    """Calculate root mean squared error from actual and predicted values."""
    return np.sqrt(np.square(t - p).mean())


# Utilities for logistic regression error

def _true_p(t, p):
    return (t & p).sum()


def _true_n(t, p):
    return (~t & ~p).sum()


def _false_p(t, p):
    return (~t & p).sum()


def _false_n(t, p):
    return (t & ~p).sum()


def accuracy(t, p):
    return np.mean(t == p)


def precision(t, p):
    return _true_p(t, p) / (_true_p(t, p) + _false_p(t, p))


def recall(t, p):
    return _true_p(t, p) / (_true_p(t, p) + _false_n(t, p))


def false_alarm(t, p):
    return _false_p(t, p) / (_false_p(t, p) + _true_n(t, p))
