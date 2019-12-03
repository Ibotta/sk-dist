"""
Utility functions for the distribute module
"""

import numbers
import numpy as np

from sklearn.metrics import check_scoring

from .validation import _safe_indexing

def _aggregate_score_dicts(scores):
    """ Aggregate the list of dict to dict of np ndarray """
    return {key: np.asarray([score[key] for score in scores])
            for key in scores[0]}

def _multimetric_score(estimator, X_test, y_test, scorers):
    """ Return a dict of score for multimetric scoring """
    scores = {}

    for name, scorer in scorers.items():
        if y_test is None:
            score = scorer(estimator, X_test)
        else:
            score = scorer(estimator, X_test, y_test)

        if hasattr(score, 'item'):
            try:
                # e.g. unwrap memmapped scalars
                score = score.item()
            except ValueError:
                # non-scalar?
                pass
        scores[name] = score

        if not isinstance(score, numbers.Number):
            raise ValueError("scoring must return a number, got %s (%s) "
                             "instead. (scorer=%s)"
                             % (str(score), type(score), name))
    return scores

def _score(estimator, X_test, y_test, scorer, is_multimetric=False):
    """
    Compute the score(s) of an estimator on a given test set. Will return
    a single float if is_multimetric is False and a dict of floats,
    if is_multimetric is True
    """
    if is_multimetric:
        return _multimetric_score(estimator, X_test, y_test, scorer)
    else:
        if y_test is None:
            score = scorer(estimator, X_test)
        else:
            score = scorer(estimator, X_test, y_test)

        if hasattr(score, 'item'):
            try:
                # e.g. unwrap memmapped scalars
                score = score.item()
            except ValueError:
                # non-scalar?
                pass

        if not isinstance(score, numbers.Number):
            raise ValueError("scoring must return a number, got %s (%s) "
                             "instead. (scorer=%r)"
                             % (str(score), type(score), scorer))
    return score

def _check_multimetric_scoring(estimator, scoring=None):
    """ Check the scoring parameter in cases when multiple metrics are allowed """
    if callable(scoring) or scoring is None or isinstance(scoring,
                                                          str):
        scorers = {"score": check_scoring(estimator, scoring=scoring)}
        return scorers, False
    else:
        err_msg_generic = ("scoring should either be a single string or "
                           "callable for single metric evaluation or a "
                           "list/tuple of strings or a dict of scorer name "
                           "mapped to the callable for multiple metric "
                           "evaluation. Got %s of type %s"
                           % (repr(scoring), type(scoring)))

        if isinstance(scoring, (list, tuple, set)):
            err_msg = ("The list/tuple elements must be unique "
                       "strings of predefined scorers. ")
            invalid = False
            try:
                keys = set(scoring)
            except TypeError:
                invalid = True
            if invalid:
                raise ValueError(err_msg)

            if len(keys) != len(scoring):
                raise ValueError(err_msg + "Duplicate elements were found in"
                                 " the given list. %r" % repr(scoring))
            elif len(keys) > 0:
                if not all(isinstance(k, str) for k in keys):
                    if any(callable(k) for k in keys):
                        raise ValueError(err_msg +
                                         "One or more of the elements were "
                                         "callables. Use a dict of score name "
                                         "mapped to the scorer callable. "
                                         "Got %r" % repr(scoring))
                    else:
                        raise ValueError(err_msg +
                                         "Non-string types were found in "
                                         "the given list. Got %r"
                                         % repr(scoring))
                scorers = {scorer: check_scoring(estimator, scoring=scorer)
                           for scorer in scoring}
            else:
                raise ValueError(err_msg +
                                 "Empty list was given. %r" % repr(scoring))

        elif isinstance(scoring, dict):
            keys = set(scoring)
            if not all(isinstance(k, str) for k in keys):
                raise ValueError("Non-string types were found in the keys of "
                                 "the given dict. scoring=%r" % repr(scoring))
            if len(keys) == 0:
                raise ValueError("An empty dict was passed. %r"
                                 % repr(scoring))
            scorers = {key: check_scoring(estimator, scoring=scorer)
                       for key, scorer in scoring.items()}
        else:
            raise ValueError(err_msg_generic)
        return scorers, True

def _num_samples(x):
    """ Return number of samples in array-like x """
    if hasattr(x, 'fit') and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError('Expected sequence or array-like, got '
                        'estimator %s' % x)
    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError("Expected sequence or array-like, got %s" %
                            type(x))
    if hasattr(x, 'shape'):
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        # Check that shape is returning an integer or default to len
        # Dask dataframes may not return numeric shape[0] value
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]
        else:
            return len(x)
    else:
        return len(x)

def _safe_split(estimator, X, y, indices, train_indices=None):
    """Create subset of dataset and properly handle kernels"""
    from sklearn.gaussian_process.kernels import Kernel as GPKernel

    if (hasattr(estimator, 'kernel') and callable(estimator.kernel) and
            not isinstance(estimator.kernel, GPKernel)):
        # cannot compute the kernel values with custom function
        raise ValueError("Cannot use a custom kernel function. "
                         "Precompute the kernel matrix instead.")

    if not hasattr(X, "shape"):
        if getattr(estimator, "_pairwise", False):
            raise ValueError("Precomputed kernels or affinity matrices have "
                             "to be passed as arrays or sparse matrices.")
        X_subset = [X[index] for index in indices]
    else:
        if getattr(estimator, "_pairwise", False):
            # X is a precomputed square kernel matrix
            if X.shape[0] != X.shape[1]:
                raise ValueError("X should be a square kernel matrix")
            if train_indices is None:
                X_subset = X[np.ix_(indices, indices)]
            else:
                X_subset = X[np.ix_(indices, train_indices)]
        else:
            X_subset = _safe_indexing(X, indices)

    if y is not None:
        y_subset = _safe_indexing(y, indices)
    else:
        y_subset = None
    return X_subset, y_subset

def _dict_slice_remove(a, b, cols=["model_index", "param_set"]):
    """ 
    Remove dictionary records from list 'a' 
    if the records appear in 'b' 
    for the given columns 'cols'.
    """
    b_ = [{k:d[k] for k in cols} for d in b]
    a_ = []
    for d in a:
        if {k:d[k] for k in cols} not in b_:
            a_.append(d)
    return a_

