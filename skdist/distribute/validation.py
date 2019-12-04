"""
Validation functions for the distribute module
"""

import numbers
import numpy as np

from itertools import compress
from scipy.sparse import issparse
from sklearn.model_selection import ParameterGrid
from sklearn.utils.validation import check_is_fitted

def _check_estimator(estimator, verbose=False):
    """ Print sparkContext awareness if apporpriate """
    if verbose:
        if estimator.sc is None:
            print("No spark context is provided; running locally")
        else:
            print("Spark context found; running with spark")

def _check_is_fitted(estimator, attributes=None):
    from sklearn import __version__
    if __version__ < '0.22':
        return check_is_fitted(estimator, attributes)
    else:
        return check_is_fitted(estimator)

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

def _is_arraylike(x):
    """ Returns whether the input is array-like """
    return (hasattr(x, '__len__') or
            hasattr(x, 'shape') or
            hasattr(x, '__array__'))

def _num_samples(x):
    """ Return number of samples in array-like x """
    message = 'Expected sequence or array-like, got %s' % type(x)
    if hasattr(x, 'fit') and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError(message)

    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, 'shape') and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        # Check that shape is returning an integer or default to len
        # Dask dataframes may not return numeric shape[0] value
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]
    try:
        return len(x)
    except TypeError:
        raise TypeError(message)

def _safe_indexing(X, indices, axis=0):
    """ Return rows, items or columns of X using indices """
    if indices is None:
        return X

    if axis not in (0, 1):
        raise ValueError(
            "'axis' should be either 0 (to index rows) or 1 (to index "
            " column). Got {} instead.".format(axis)
        )

    indices_dtype = _determine_key_type(indices)

    if axis == 0 and indices_dtype == 'str':
        raise ValueError(
            "String indexing is not supported with 'axis=0'"
        )

    if axis == 1 and X.ndim != 2:
        raise ValueError(
            "'X' should be a 2D NumPy array, 2D sparse matrix or pandas "
            "dataframe when indexing the columns (i.e. 'axis=1'). "
            "Got {} instead with {} dimension(s).".format(type(X), X.ndim)
        )

    if axis == 1 and indices_dtype == 'str' and not hasattr(X, 'loc'):
        raise ValueError(
            "Specifying the columns using strings is only supported for "
            "pandas DataFrames"
        )

    if hasattr(X, "iloc"):
        return _pandas_indexing(X, indices, indices_dtype, axis=axis)
    elif hasattr(X, "shape"):
        return _array_indexing(X, indices, indices_dtype, axis=axis)
    else:
        return _list_indexing(X, indices, indices_dtype)

def _determine_key_type(key):
    """ Determine the data type of key """
    err_msg = ("No valid specification of the columns. Only a scalar, list or "
               "slice of all integers or all strings, or boolean mask is "
               "allowed")

    dtype_to_str = {int: 'int', str: 'str', bool: 'bool', np.bool_: 'bool'}
    array_dtype_to_str = {'i': 'int', 'u': 'int', 'b': 'bool', 'O': 'str',
                          'U': 'str', 'S': 'str'}

    if key is None:
        return None
    if isinstance(key, tuple(dtype_to_str.keys())):
        try:
            return dtype_to_str[type(key)]
        except KeyError:
            raise ValueError(err_msg)
    if isinstance(key, slice):
        if key.start is None and key.stop is None:
            return None
        key_start_type = _determine_key_type(key.start)
        key_stop_type = _determine_key_type(key.stop)
        if key_start_type is not None and key_stop_type is not None:
            if key_start_type != key_stop_type:
                raise ValueError(err_msg)
        if key_start_type is not None:
            return key_start_type
        return key_stop_type
    if isinstance(key, list):
        unique_key = set(key)
        key_type = {_determine_key_type(elt) for elt in unique_key}
        if not key_type:
            return None
        if len(key_type) != 1:
            raise ValueError(err_msg)
        return key_type.pop()
    if hasattr(key, 'dtype'):
        try:
            return array_dtype_to_str[key.dtype.kind]
        except KeyError:
            raise ValueError(err_msg)
    raise ValueError(err_msg)

def _pandas_indexing(X, key, key_dtype, axis):
    """ Index a pandas dataframe or a series """
    if hasattr(key, 'shape'):
        # Work-around for indexing with read-only key in pandas
        # FIXME: solved in pandas 0.25
        key = np.asarray(key)
        key = key if key.flags.writeable else key.copy()
    # check whether we should index with loc or iloc
    indexer = X.iloc if key_dtype == 'int' else X.loc
    return indexer[:, key] if axis else indexer[key]

def _list_indexing(X, key, key_dtype):
    """ Index a Python list """
    if np.isscalar(key) or isinstance(key, slice):
        # key is a slice or a scalar
        return X[key]
    if key_dtype == 'bool':
        # key is a boolean array-like
        return list(compress(X, key))
    # key is a integer array-like of key
    return [X[idx] for idx in key]

def _array_indexing(array, key, key_dtype, axis):
    """ Index an array or scipy.sparse consistently across NumPy version """
    if issparse(array):
        if key_dtype == 'bool':
            key = np.asarray(key)
    return array[key] if axis == 0 else array[:, key]
