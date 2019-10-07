"""
Validation functions for the distribute module
"""

import numpy as np

from sklearn.model_selection import ParameterGrid

def _check_estimator(estimator, verbose=False):
    """ Print sparkContext awareness if apporpriate """
    if verbose:
        if estimator.sc is None:
            print("No spark context is provided; running locally")
        else:
            print("Spark context found; running with spark")

def _validate_params(param_sets):
    """ Validates grid params """
    invalid_params = [
        type(param) 
        for param in param_sets 
        if not isinstance(param, dict)
        ]
    if invalid_params:
        raise ValueError('Params must be dictionaries: got '
            '{0!r}'.format(invalid_params))  

def _validate_models(models, clf):
    """ Validates models input argument """
    try:
        iter(models)
    except:
        raise TypeError("Input argument 'models' is not iterable.")
    if not (isinstance(models[0], tuple) or isinstance(models[0], list)):
        models = [models]
    names, estimators, param_sets = zip(*models)
    _validate_names(clf, names)
    _validate_estimators(estimators)
    _validate_params(param_sets)
    return models

def _check_base_estimator(estimator):
    """ Validates base estimator """
    return (
        hasattr(estimator, "fit") and 
        hasattr(estimator, "predict")
        )
        
def _validate_estimators(estimators):
    """ Validates estimators """
    invalid_estimators = [
        type(estimator) 
        for estimator in estimators 
        if not _check_base_estimator(estimator)
        ]
    if invalid_estimators:
        raise ValueError('Estimators must be sklearn estimators: got '
            '{0!r}'.format(invalid_estimators))

def _validate_names(cls, names):
    """ Validates names """
    if len(set(names)) != len(names):
        raise ValueError('Names provided are not unique: '
            '{0!r}'.format(list(names)))
    invalid_names = [name for name in names if not isinstance(name, str)]
    if invalid_names:
        raise ValueError('Estimator names must be strings: got '
            '{0!r}'.format(invalid_names))
    invalid_names = set(names).intersection(cls.get_params(deep=False))
    if invalid_names:
        raise ValueError('Estimator names conflict with constructor '
            'arguments: {0!r}'.format(sorted(invalid_names)))
    invalid_names = [name for name in names if '__' in name]
    if invalid_names:
        raise ValueError('Estimator names must not contain __: got '
            '{0!r}'.format(invalid_names))

def _check_n_iter(n_iter, param_distributions):
    """
    Check if n_iter is greater than the total number 
    of possible param sets from the given distribution.
    """
    all_lists = np.all([not hasattr(v, "rvs") 
        for v in param_distributions.values()])
    if all_lists:
        param_grid = ParameterGrid(param_distributions)
        grid_size = len(param_grid)
    else:
        grid_size = n_iter
    return min(grid_size, n_iter)
