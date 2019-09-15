"""
Distributed multiclass meta-estimators
"""

import numpy as np
import pandas as pd
import warnings

from joblib import Parallel, delayed
from copy import copy
from collections.abc import Sequence
from scipy.sparse import vstack
from sklearn.base import BaseEstimator
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
from sklearn.utils.validation import check_X_y
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.model_selection import train_test_split, GroupKFold

from .validation import _check_estimator
from .utils import _safe_split
from .base import (
    _clone, _get_value, _parse_partitions
    )

__all__ = [
    "DistOneVsRestClassifier",
    "DistOneVsOneClassifier"
]

def _chunks(l, n):
    """ Yield successive n-sized chunks from input list """
    for i in range(0, len(l), n):
        yield l[i:i + n]

def _split_X(X, n_splits, sc):
    """ Split and broadcast a sparse matrix """
    if n_splits > 1:
        if isinstance(X, pd.DataFrame):
            inds = list(range(len(X)))
            inds_list = list(_chunks(inds, int(len(inds) / float(n_splits))))
            Xs = [X.ix[slc] for slc in inds_list]
        else:
            # X is a scipy sparse matrix
            inds = list(range(X.shape[0]))
            inds_list = list(_chunks(inds, int(len(inds) / float(n_splits))))
            Xs = [X[slc, :] for slc in inds_list]

        return [sc.broadcast(x) for x in Xs]
    else:
        return X

def _combine_Xs(Xs):
    """ Combine and access values of broadcasted, splitted sparse matrix """
    if isinstance(Xs, list):
        if isinstance(Xs[0].value, pd.DataFrame):
            df = pd.DataFrame()
            return df.append([x.value for x in Xs])
        else:
            return vstack([x.value for x in Xs])
    else:
        return Xs

def _use_best_estimator(x):
    """ Pull best estimator out of meta-estimator if available """
    estimator_ = x.best_estimator_ if hasattr(x, "best_estimator_") else x
    if hasattr(x, "cv_results_"):
        estimator_.cv_results_ = pd.DataFrame(x.cv_results_)
        for col in estimator_.cv_results_.columns:
            estimator_.cv_results_[col] = estimator_.cv_results_[col].astype(str)
        estimator_.cv_results_ = estimator_.cv_results_.to_dict("list")
    return estimator_

def _negatives_mask(X, y, max_negatives=None, random_state=None, method="ratio"):
    """ Limit the number of negative records in training set """
    if max_negatives is None:
        return [X, y]
    else:
        pos_mask = y == 1
        if method == "ratio":
            pass
        elif method == "multiplier":
            max_negatives = int(max_negatives * len(y[pos_mask]))
        else:
            raise ValueError("Unknown method. Options are 'ratio' or 'multiplier'.")
        if isinstance(max_negatives, int) and max_negatives >= len(y[~pos_mask]):
            return [X, y]
        max_negatives = (
            max_negatives
            if isinstance(max_negatives, float)
            else (max_negatives / float(len(y[~pos_mask]))))
        _, X_neg, _, y_neg = train_test_split(
            X[~pos_mask, :], y[~pos_mask],
            test_size=max_negatives, random_state=random_state)
        stack_func = vstack if hasattr(X, "nnz") else np.vstack
        return shuffle(
            stack_func([X[pos_mask, :], X_neg]), np.concatenate([y[pos_mask], y_neg]),
            random_state=random_state)

def _fit_binary(estimator, X, y, fit_params, classes=None, max_negatives=None, random_state=None, method="ratio"):
    """ Fit a single binary estimator """
    X = _combine_Xs(X)
    unique_y = np.unique(y)
    if len(unique_y) == 1:
        if classes is not None:
            if y[0] == -1:
                c = 0
            else:
                c = y[0]
            warnings.warn("Label %s is present in all training examples." %
                          str(classes[c]))
        est = _ConstantPredictor().fit(*_negatives_mask(
            (X.value if hasattr(X, "value") else X), unique_y,
            max_negatives=max_negatives, random_state=random_state,
            method=method))
    else:
        est = _clone(_get_value(estimator))
        est.fit(*_negatives_mask(
            (X.value if hasattr(X, "value") else X), y,
            max_negatives=max_negatives, random_state=random_state,
            method=method), **fit_params)
    return _use_best_estimator(est), classes[1]

def _fit_ovo_binary(estimator, X, y, i, j, fit_params):
    """ Fit a single binary estimator (one-vs-one) """
    cond = np.logical_or(y == i, y == j)
    y_ = y[cond]
    y_binary = np.empty(y_.shape, np.int)
    y_binary[y_ == i] = 0
    y_binary[y_ == j] = 1
    indcond = np.arange(X.shape[0])[cond]
    return _fit_binary(estimator,
                       _safe_split(estimator, X, None, indices=indcond)[0],
                       y_binary, fit_params, classes=[i, j]), indcond

class _ConstantPredictor(BaseEstimator):
    """ Predicts same labels as trained """ 
    def fit(self, X, y):
        self.y_ = y
        return self

    def predict(self, X):
        check_is_fitted(self, 'y_')
        return np.repeat(self.y_, X.shape[0])

    def decision_function(self, X):
        check_is_fitted(self, 'y_')
        return np.repeat(self.y_, X.shape[0])

    def predict_proba(self, X):
        check_is_fitted(self, 'y_')
        return np.repeat([np.hstack([1 - self.y_, self.y_])],
                         X.shape[0], axis=0)

class DistOneVsRestClassifier(OneVsRestClassifier):
    """
    Same as sklearn `OneVsRestClassifier` but with distributed
    training using spark. Additionally implements flexible
    ``predict_proba`` method with custom `norm` input
    designating the normalization method used after individual
    predictions are made.

    Args:
        estimator (sklearn estimator): An estimator object implementing fit and one of
            decision_function or predict_proba.
        sc (sparkContext): Spark context for spark broadcasting and rdd operations.
        norm (string): default None, Normalization method for predict_proba.
        partitions (int or 'auto'): default 'auto'
            Number of partitions to use for parallelization of parameter
            search space. Integer values or None will be used directly for `numSlices`,
            while 'auto' will set `numSlices` to the number required fits.
        max_negatives (int or float): default None
            Maximum number of negative records allowed for each binary
            estimator. Use int for hard maximum, or float for percentage
            of total negatives.
        random_state (int): default None
            Random state for limiting negatives (if max_negatives is not None).
        method (str): 'ratio' or 'multiplier'
            Method used to calculate true maximum number of negatives.
        n_splits (int): default 1
            Dials the number of splits for broadcasting
            X during fitting. Use values higher than 1 for large X.
        mlb_override (bool): pass over mlb step; this assumes
            that input `y` to `fit` is already in sparse (one-hot-encoded)
            format
        verbose (bool): print status messages
        **kwargs: Keyword arguments to be passed to `OneVsRestClassifier`.
    """
    def __init__(self, estimator, sc=None, norm=None, partitions='auto',
            max_negatives=None, random_state=None, method="ratio",
            n_splits=1, mlb_override=False, verbose=False, **kwargs):
        OneVsRestClassifier.__init__(
            self, estimator, **kwargs)
        self.norm = norm
        self.sc = sc
        self.partitions = partitions
        self.max_negatives = max_negatives
        self.random_state = random_state
        self.method = method
        self.n_splits = n_splits
        self.mlb_override = mlb_override
        self.verbose = verbose

    def fit(self, X, y, **fit_params):
        """
        Fit underlying estimators. Parallelize fit operation using spark.

        Args:
            X (array-like, shape = [n_samples, n_features]): input data
            y (array-like, shape = [n_samples, ], [n_samples, n_classes]): multi-class targets
            **fit_params (dict of string -> object): parameters passed 
                to the ``fit`` method of the estimator
        """
        _check_estimator(self, verbose=self.verbose)

        if (not self.mlb_override and not hasattr(y[0], '__array__') 
                and isinstance(y[0], Sequence)
                and not isinstance(y[0], str)):
            self.mlb = MultiLabelBinarizer()
            y = self.mlb.fit_transform(y)

        if isinstance(X, pd.DataFrame):
            X.index = list(range(len(X)))

        self.label_binarizer_ = LabelBinarizer(sparse_output=True)
        self.label_binarizer_.fit(y)
        self.classes_ = self.label_binarizer_.classes_
        self._fit(X, y, **fit_params)
        del self.sc
        if hasattr(self.estimator, "sc"):
            del self.estimator.sc
        return self

    def _fit(self, X, y, **fit_params):
        Y = self.label_binarizer_.transform(y)
        Y = Y.tocsc()
        max_negatives = self.max_negatives
        random_state = self.random_state
        n_splits = self.n_splits
        method = self.method
        estimator = _clone(self.estimator)
        if self.sc is None:
            models_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_binary)(
                    estimator, X, x[1], fit_params, 
                    classes=["not %s" % x[0], x[0]],
                    max_negatives=max_negatives, 
                    random_state=random_state, method=method)
                for x in list(zip(self.classes_, list(col.toarray().ravel() for col in Y.T))))
        else:
            X = _split_X(X, n_splits, self.sc)
            partitions = _parse_partitions(self.partitions, len(self.classes_))
            estimator = self.sc.broadcast(self.estimator)
            columns = self.sc.parallelize(
                list(zip(self.classes_, list(col.toarray().ravel() for col in Y.T))),
                numSlices=partitions)
            models_ = columns.map(lambda x: _fit_binary(
                estimator, X, x[1], fit_params, classes=["not %s" % x[0], x[0]],
                max_negatives=max_negatives, random_state=random_state, method=method)).collect()
        estimators_ = [x[0] for x in models_]
        classes_ = [x[1] for x in models_]
        self.estimators_ = list([estimators_[classes_.index(x)] for x in self.classes_])
        return self

    def predict_proba(self, X):
        """
        Probability estimates.
        The returned estimates for all classes are ordered by label of classes.

        Args:
            X (array-like, shape = [n_samples, n_features]): input data

        Returns:
            T (array-like, shape = [n_samples, n_classes]): returns the probability 
                of the sample for each class in the model, where classes are 
                ordered as they are in self.classes_
        """
        probs = []
        for index in range(len(self.estimators_)):
            probs.append(self.estimators_[index].predict_proba(X)[:,1])
        out = np.array([
            [probs[y][index] for y in range(len(self.estimators_))]
            for index in range(len(probs[0]))])
        if self.norm:
            return normalize(out, norm=self.norm)
        else:
            return out

class DistOneVsOneClassifier(OneVsOneClassifier):
    """
    Same as sklearn `OneVsOneClassifier` but with distributed
    training using spark.

    Args:
        estimator (sklearn estimator): An estimator object implementing fit and one of
            decision_function or predict_proba.
        sc (sparkContext): Spark context for spark broadcasting and rdd operations.
        partitions (int or 'auto'): default 'auto'
            Number of partitions to use for parallelization of parameter
            search space. Integer values or None will be used directly for `numSlices`,
            while 'auto' will set `numSlices` to the number required fits.
        verbose (bool): print status messages
        **kwargs: Keyword arguments to be passed to `OneVsOneClassifier`.
    """
    def __init__(self, estimator, sc=None, partitions='auto', verbose=False, **kwargs):
        OneVsOneClassifier.__init__(
            self, estimator, **kwargs)
        self.sc = sc
        self.partitions = partitions
        self.verbose = verbose

    def fit(self, X, y, **fit_params):
        """
        Fit underlying estimators. Parallelize fit operation using spark.

        Args:
            X (array-like, shape = [n_samples, n_features]): intput
                data
            y (array-like, shape = [n_samples]): Multi-class targets.
            **fit_params (dict of string -> object): parameters passed 
                to the ``fit`` method of the estimator
        """
        _check_estimator(self, verbose=self.verbose)

        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'])
        check_classification_targets(y)

        self.classes_ = np.unique(y)
        if len(self.classes_) == 1:
            raise ValueError("OneVsOneClassifier can not be fit when only one"
                             " class is present.")
        n_classes = self.classes_.shape[0]
        class_list = []
        count = -1
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                count += 1
                class_list.append([count, self.classes_[i], self.classes_[j]])
        estimator = _clone(self.estimator)

        if self.sc is None:
            estimators_indices = list(zip(*(Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_ovo_binary)
                (self.estimator, X, y, self.classes_[i], self.classes_[j], fit_params)
                for i in range(n_classes) for j in range(i + 1, n_classes)))))
            self.estimators_ = [x[0] for x in estimators_indices[0]]
            self.pairwise_indices_ = (
                estimators_indices[1] if self._pairwise else None)
        else:
            estimator = self.sc.broadcast(self.estimator)
            partitions = _parse_partitions(self.partitions, len(class_list))
            estimators_indices = self.sc.parallelize(
                    class_list,
                    numSlices=partitions).groupBy(lambda x: x[0]).map(
                    lambda x: [x[0], _fit_ovo_binary(estimator, X, y, list(x[1])[0][1], list(x[1])[0][2], fit_params)]).collect()

            estimators_indices = [estimators_indices[i][1] for i in np.argsort([x[0] for x in estimators_indices])]
            self.estimators_ = [x[0][0] for x in estimators_indices]
            try:
                self.pairwise_indices_ = (
                    [x[1] for x in estimators_indices] if self._pairwise else None)
            except AttributeError:
                self.pairwise_indices_ = None
        del self.sc
        if hasattr(self.estimator, "sc"):
            del self.estimator.sc
        return self
