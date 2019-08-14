"""
Distributed ensemble meta-estimators
"""

import numpy as np
import pandas as pd

from scipy.sparse import hstack, issparse
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble.forest import (
    BaseForest, ExtraTreesClassifier,
    ExtraTreesRegressor, ForestClassifier,
    ForestRegressor, RandomForestClassifier,
    RandomForestRegressor
    )
from sklearn.model_selection import KFold
from sklearn.tree.tree import DOUBLE, DTYPE
from sklearn.utils import check_array, check_random_state

from .base import _check_estimator

__all__ = [
    "DistRandomForestClassifier",
    "DistRandomForestRegressor",
    "DistExtraTreesClassifier",
    "DistExtraTreesRegressor"
]

MAX_RAND_SEED = np.iinfo(np.int32).max

def _set_random_states(estimator, random_state=None):
    """Sets fixed random_state parameters for an estimator"""
    random_state = check_random_state(random_state)
    to_set = {}
    for key in sorted(estimator.get_params(deep=True)):
        if key == 'random_state' or key.endswith('__random_state'):
            to_set[key] = random_state.randint(MAX_RAND_SEED)
    if to_set:
        estimator.set_params(**to_set)

def _generate_sample_indices(random_state, n_samples):
    """Private function used to _parallel_build_trees function."""
    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples)
    return sample_indices

def _make_estimator(base_estimator, estimator_params, params={}, random_state=None):
    """Make and configure a copy of `base_estimator`"""
    estimator = clone(base_estimator)
    estimator.set_params(**dict((p, params[p])
                                for p in estimator_params))

    if random_state is not None:
        _set_random_states(estimator, random_state)
    return estimator

def _build_trees(base_estimator, estimator_params, params, X, y, sample_weight,
                          tree_idx, n_trees, verbose=0, class_weight=None,
                          random_state=None, bootstrap=False):
    """Fit a single tree in parallel"""
    tree = _make_estimator(
        base_estimator.value, estimator_params,
        params=params, random_state=random_state)
    if verbose > 1:
        print(("building tree %d of %d" % (tree_idx + 1, n_trees)))

    if bootstrap:
        n_samples = X.shape[0]
        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
        else:
            curr_sample_weight = sample_weight.copy()

        indices = _generate_sample_indices(random_state, n_samples)
        sample_counts = np.bincount(indices, minlength=n_samples)
        curr_sample_weight *= sample_counts

        if class_weight == 'subsample':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', DeprecationWarning)
                curr_sample_weight *= compute_sample_weight('auto', y, indices)
        elif class_weight == 'balanced_subsample':
            curr_sample_weight *= compute_sample_weight('balanced', y, indices)

        tree.fit(X, y, sample_weight=curr_sample_weight, check_input=False)
    else:
        tree.fit(X, y, sample_weight=sample_weight, check_input=False)
    return tree

def get_single_oof(clf, X, y, train_index, test_index):
    """Fit on the data specified by the train_index and predict_proba on the
    test index"""
    if isinstance(X, (pd.DataFrame, pd.Series)):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
    else:
        X_train = X[train_index]
        X_test = X[test_index]

    y_train = y[train_index]

    clf.fit(X_train, y_train)
    return test_index, clf.predict_proba(X_test)

def get_oof(clf, X, y, n_splits=5):
    """Fit the classifier to the data and make predictions out of fold."""
    kfold = KFold(n_splits=n_splits)

    oof_train = np.zeros((y.shape[0], len(np.unique(y))))

    for train_index, test_index in kfold.split(X):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
        else:
            X_train = X[train_index]
            X_test = X[test_index]

        y_train = y[train_index]

        clf.fit(X_train, y_train)

        oof_train[test_index] = clf.predict_proba(X_test)

    clf.fit(X, y)
    return clf, oof_train

class DistBaseForest(BaseForest):
    """
    Same as sklearn `BaseForest` but with distributed
    training using spark.

    Args:
        base_estimator (sklearn estimator): An estimator object implementing fit and one of
            decision_function or predict_proba.
        sc (sparkContext): Spark context for spark broadcasting and rdd operations.
        partitions (int or 'auto'): default 'auto'
            Number of partitions to use for parallelization of parameter
            search space. Integer values or None will be used directly for `numSlices`,
            while 'auto' will set `numSlices` to the number required fits.
        **kwargs: Keyword arguments to be passed to `BaseForest`.
    """
    def __init__(self, base_estimator, sc=None, partitions='auto', **kwargs):
        BaseForest.__init__(
            self, base_estimator, **kwargs)
        self.sc = sc
        self.partitions = partitions

    def fit(self, X, y, sample_weight=None):
        """
        Build a forest of trees from the training set (X, y). Parallelize fit
        operation using spark.

        Args:
            X : array-like or sparse matrix of shape = [n_samples, n_features]
                The training input samples. Internally, its dtype will be converted to
                ``dtype=np.float32``. If a sparse matrix is provided, it will be
                converted into a sparse ``csc_matrix``.
            y : array-like, shape = [n_samples] or [n_samples, n_outputs]
                The target values (class labels in classification, real numbers in
                regression).
            sample_weight : array-like, shape = [n_samples] or None
                Sample weights. If None, then samples are equally weighted. Splits
                that would create child nodes with net zero or negative weight are
                ignored while searching for a split in each node. In the case of
                classification, splits are also ignored if they would result in any
                single class carrying a negative weight in either child node.
        """
        _check_estimator(self)        

        # Validate or convert input data
        X = check_array(X, accept_sparse="csc", dtype=DTYPE)
        y = check_array(y, accept_sparse='csc', ensure_2d=False, dtype=None)
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False)
        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        # Remap output
        n_samples, self.n_features_ = X.shape

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn("A column-vector y was passed when a 1d array was"
                 " expected. Please change the shape of y to "
                 "(n_samples,), for example using ravel().",
                 DataConversionWarning, stacklevel=2)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        y, expanded_class_weight = self._validate_y_class_weight(y)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Check parameters
        self._validate_estimator()

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available"
                             " if bootstrap=True")

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError('n_estimators=%d must be larger or equal to '
                             'len(estimators_)=%d when warm_start==True'
                             % (self.n_estimators, len(self.estimators_)))

        elif n_more_estimators == 0:
            warn("Warm-start fitting without increasing n_estimators does not "
                 "fit new trees.")
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            base_estimator = self.sc.broadcast(self.base_estimator_)
            estimator_params = self.estimator_params
            verbose = self.verbose
            class_weight = self.class_weight
            bootstrap = self.bootstrap
            rs = self.random_state
            params = {}
            for p in self.estimator_params:
                params[p] = getattr(self, p)
            if self.partitions is None:
                partitions = None
            elif self.partitions == 'auto':
                partitions = n_more_estimators
            else:
                try:
                    partitions = int(self.partitions)
                except:
                    partitions = None
            trees = self.sc.parallelize(
                list(range(n_more_estimators)),
                numSlices=partitions).map(
                lambda x: _build_trees(
                    base_estimator, estimator_params, params, X, y, sample_weight,
                    x, n_more_estimators, verbose=verbose,
                    class_weight=class_weight, random_state=rs,
                    bootstrap=bootstrap)).collect()

            # Collect newly grown trees
            self.estimators_.extend(trees)

        if self.oob_score:
            self._set_oob_score(X, y)

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        del self.sc
        return self

    def _set_oob_score(self, X, y):
        """Calculate out of bag predictions and score"""
        pass

class DistForestClassifier(DistBaseForest, ForestClassifier):
    """
    Same as sklearn `ForestClassifier` but with distributed
    training using spark.

    Args:
        base_estimator (sklearn estimator): An estimator object implementing fit and one of
            decision_function or predict_proba.
        sc (sparkContext): Spark context for spark broadcasting and rdd operations.
        partitions (int or 'auto'): default 'auto'
            Number of partitions to use for parallelization of parameter
            search space. Integer values or None will be used directly for `numSlices`,
            while 'auto' will set `numSlices` to the number required fits.
        **kwargs: Keyword arguments to be passed to `ForestClassifier`.
    """
    def __init__(self, base_estimator, sc=None, partitions='auto', **kwargs):
        ForestClassifier.__init__(
            self, base_estimator, **kwargs)
        self.sc = sc
        self.partitions = partitions

class DistRandomForestClassifier(DistForestClassifier, RandomForestClassifier):
    """
    Same as sklearn `RandomForestClassifier` but with distributed
    training using spark.

    Args:
        base_estimator (sklearn estimator): An estimator object implementing fit and one of
            decision_function or predict_proba.
        sc (sparkContext): Spark context for spark broadcasting and rdd operations.
        partitions (int or 'auto'): default 'auto'
            Number of partitions to use for parallelization of parameter
            search space. Integer values or None will be used directly for `numSlices`,
            while 'auto' will set `numSlices` to the number required fits.
        **kwargs: Keyword arguments to be passed to `RandomForestClassifier`.
    """
    def __init__(self, sc=None, partitions='auto', **kwargs):
        RandomForestClassifier.__init__(
            self, **kwargs)
        self.sc = sc
        self.partitions = partitions

class DistExtraTreesClassifier(DistForestClassifier, ExtraTreesClassifier):
    """
    Same as sklearn `ExtraTreesClassifier` but with distributed
    training using spark.

    Args:
        base_estimator (sklearn estimator): An estimator object implementing fit and one of
            decision_function or predict_proba.
        sc (sparkContext): Spark context for spark broadcasting and rdd operations.
        partitions (int or 'auto'): default 'auto'
            Number of partitions to use for parallelization of parameter
            search space. Integer values or None will be used directly for `numSlices`,
            while 'auto' will set `numSlices` to the number required fits.
        **kwargs: Keyword arguments to be passed to `ExtraTreesClassifier`.
    """
    def __init__(self, sc=None, partitions='auto', **kwargs):
        ExtraTreesClassifier.__init__(
            self, **kwargs)
        self.sc = sc
        self.partitions = partitions

class DistForestRegressor(DistBaseForest, ForestRegressor):
    """
    Same as sklearn `ForestRegressor` but with distributed
    training using spark.

    Args:
        base_estimator (sklearn estimator): An estimator object implementing fit and one of
            decision_function or predict_proba.
        sc (sparkContext): Spark context for spark broadcasting and rdd operations.
        partitions (int or 'auto'): default 'auto'
            Number of partitions to use for parallelization of parameter
            search space. Integer values or None will be used directly for `numSlices`,
            while 'auto' will set `numSlices` to the number required fits.
        **kwargs: Keyword arguments to be passed to `ForestRegressor`.
    """
    def __init__(self, base_estimator, sc=None, partitions='auto', **kwargs):
        ForestRegressor.__init__(
            self, base_estimator, **kwargs)
        self.sc = sc
        self.partitions = partitions

class DistRandomForestRegressor(DistForestRegressor, RandomForestRegressor):
    """
    Same as sklearn `RandomForestRegressor` but with distributed
    training using spark.

    Args:
        base_estimator (sklearn estimator): An estimator object implementing fit and one of
            decision_function or predict_proba.
        sc (sparkContext): Spark context for spark broadcasting and rdd operations.
        partitions (int or 'auto'): default 'auto'
            Number of partitions to use for parallelization of parameter
            search space. Integer values or None will be used directly for `numSlices`,
            while 'auto' will set `numSlices` to the number required fits.
        **kwargs: Keyword arguments to be passed to `RandomForestRegressor`.
    """
    def __init__(self, sc=None, partitions='auto', **kwargs):
        RandomForestRegressor.__init__(
            self, **kwargs)
        self.sc = sc
        self.partitions = partitions

class DistExtraTreesRegressor(DistForestRegressor, ExtraTreesRegressor):
    """
    Same as sklearn `ExtraTreesRegressor` but with distributed
    training using spark.

    Args:
        base_estimator (sklearn estimator): An estimator object implementing fit and one of
            decision_function or predict_proba.
        sc (sparkContext): Spark context for spark broadcasting and rdd operations.
        partitions (int or 'auto'): default 'auto'
            Number of partitions to use for parallelization of parameter
            search space. Integer values or None will be used directly for `numSlices`,
            while 'auto' will set `numSlices` to the number required fits.
        **kwargs: Keyword arguments to be passed to `ExtraTreesRegressor`.
    """
    def __init__(self, sc=None, partitions='auto', **kwargs):
        ExtraTreesRegressor.__init__(
            self, **kwargs)
        self.sc = sc
        self.partitions = partitions

