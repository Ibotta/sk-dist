"""
Feature selection models
"""

import numpy as np

from sklearn.base import (
    BaseEstimator, ClassifierMixin, 
    is_classifier
    )
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils import check_X_y, safe_sqr
from sklearn.model_selection import check_cv
from sklearn.metrics import check_scoring
from joblib import Parallel, delayed
from itertools import product
from scipy.sparse import issparse

from .validation import _check_is_fitted
from .base import _parse_partitions, _clone
from .utils import _safe_split

__all__ = ["DistFeatureEliminator"]

def _drop_col(X, index):
    """ Drop index columns from numpy array or sparse matrix """
    cols = np.arange(X.shape[1])
    cols_to_keep = np.where(np.logical_not(np.in1d(cols, index)))[0]
    return X[:, cols_to_keep]

def _fit_and_score_one(index, estimator, X, y, scorer, train, test, verbose, fit_params):
    """ Fit and score an estimator with one feature left out """
    estimator_ = _clone(estimator)
    X_train, y_train = _safe_split(estimator, _drop_col(X, index), y, train)
    X_test, y_test = _safe_split(estimator, _drop_col(X, index), y, test, train)
    estimator_.fit(X_train, y_train, **fit_params)
    return scorer(estimator_, X_test, y_test)

def _divide_chunks(l, n): 
    """ Divide iterable into n-sized chunks """
    for i in range(0, len(l), n):  
        yield l[i:i + n] 

class DistFeatureEliminator(BaseEstimator, ClassifierMixin):
    """
    Similar to ``sklearn.feature_selection.RFECV`` but with distributed
    feature set removal and cross validation. Fits the base estimator on 
    the input training data once for each left out feature sets according 
    to the `step` and the scoring of the features according to an initial 
    estimator fit with all features. Uses cross validation to score each 
    feature set. Results in the fitted base estimator with the best feature set.

    Roughly approximates `RFECV` particularly as `step` approaches 1 and 
    `min_features_to_select` approaches 0. The value-add is in the 
    embarrassingly parallel nature of the `DistFeatureEliminator` algorithm.
    
    Args:
        estimator (estimator object):
            This is assumed to implement the scikit-learn estimator interface.
            Either estimator needs to provide a ``score`` function,
            or ``scoring`` must be passed.
        sc (sparkContext): Spark context for spark broadcasting and rdd operations.
        partitions (int or 'auto'): default 'auto'
            Number of partitions to use for parallelization of parameter
            search space. Integer values or None will be used directly for `numSlices`,
            while 'auto' will set `numSlices` to the number required fits.
        min_features_to_select (int): Minimum number of features to keep after elimination.
        step (int or float): If greater than or equal to 1, then `step` 
            corresponds to the (integer) number of features to remove at each iteration.
            If within (0.0, 1.0), then `step` corresponds to the percentage
            (rounded down) of features to remove at each iteration.
        cv (int or cv object): Determines the cross-validation splitting strategy.
        scoring (str or callable): Scoring function or callable to evaluate 
            predictions on the training set for feature elimination.
        verbose (int or bool): Controls the verbosity: the higher, the more messages.
        n_jobs (int): Number of jobs to run in parallel if using joblib.
        pre_dispatch (bool): Controls number of jobs dispatched if using joblib.
        mask (bool): Whether or not to apply feature selection mask to input data
            before making predictions.
    """
    def __init__(self, 
                 estimator, 
                 sc=None,
                 partitions='auto',
                 min_features_to_select=None, 
                 step=1,
                 cv=5, 
                 scoring=None, 
                 verbose=False, 
                 n_jobs=None, 
                 pre_dispatch=None, 
                 mask=True):
        self.estimator = estimator
        self.sc = sc
        self.partitions = partitions
        self.min_features_to_select = min_features_to_select
        self.step = step
        self.cv = cv
        self.scoring = scoring
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.mask = mask
        
    def fit(self, X, y=None, groups=None, **fit_params):
        """ 
        Apply feature elimination routine, ultimately fitting 
        estimator on the best feature set.

        Args:
            X (array-like, shape = [n_samples, n_features]): input data
            y (array-like, shape = [n_samples, ], [n_samples, n_classes]): targets
            groups (array-like): group labels for the samples used while 
                splitting the dataset into train/test set
            **fit_params (dict of string -> object): parameters passed
                to the `fit` method of the estimator
        """
        X, y = check_X_y(X, y, "csr", ensure_min_features=2)
        cv = check_cv(self.cv, y, is_classifier(self.estimator))
        scorer = check_scoring(self.estimator, scoring=self.scoring)

        n_features = X.shape[1]
        if self.min_features_to_select is None:
            min_features_to_select = n_features // 2
        else:
            min_features_to_select = self.min_features_to_select

        if 0.0 < self.step < 1.0:
            step = int(max(1, self.step * n_features))
        else:
            step = int(self.step)
        if step <= 0:
            raise ValueError("Step must be >0")

        initial_estimator = _clone(self.estimator)
        initial_estimator.fit(X, y, **fit_params)
        if hasattr(initial_estimator, 'coef_'):
            coefs = initial_estimator.coef_
        else:
            coefs = getattr(initial_estimator, 'feature_importances_', None)
        if coefs is None:
            raise RuntimeError('The classifier does not expose '
                                '"coef_" or "feature_importances_" '
                                'attributes')
        if coefs.ndim > 1:
            ranks = np.argsort(safe_sqr(coefs).sum(axis=0))
        else:
            ranks = np.argsort(safe_sqr(coefs))
        ranks = np.ravel(ranks)[:(n_features - min_features_to_select)]

        this_step = 0
        features_to_remove = [np.array([])]
        while this_step < (n_features - min_features_to_select):
            this_step += step
            features_to_remove.append(ranks[:this_step])

        cv_splits_ = list(cv.split(X,y,groups))
        fit_sets = list(product(features_to_remove, cv_splits_))
        base_estimator = _clone(self.estimator)
        if not self.sc:
            parallel = Parallel(
                n_jobs=self.n_jobs, verbose=self.verbose, 
                pre_dispatch=self.pre_dispatch
                )
            scores = parallel(
                delayed(_fit_and_score_one)(
                    index, _clone(base_estimator), X, y, 
                    scorer, train, test, self.verbose,
                    fit_params)
                for index, (train, test) in fit_sets)
            score_sets = _divide_chunks(list(scores), len(cv_splits_))
        else:
            base_estimator_ = self.sc.broadcast(base_estimator)
            partitions = _parse_partitions(self.partitions, len(fit_sets))
            verbose = self.verbose
            scores = (
                self.sc.parallelize(fit_sets, numSlices=partitions)
                .map(lambda x: [x[0], _fit_and_score_one(
                    x[0], _clone(base_estimator), X, y, scorer, 
                    x[1][0], x[1][1], verbose, fit_params)]).collect()
                )
            score_sets = []
            for feat_set in features_to_remove:
                this_set = []
                for row in scores:
                    if (feat_set.shape == row[0].shape) and np.allclose(feat_set, row[0]):
                        this_set.append(row[1])
                score_sets.append(this_set)
            
        self.scores_ = []
        for score_set in score_sets:
            this_score = np.mean(score_set)
            self.scores_.append(this_score)
            
        best_set_ = np.argmax(self.scores_)
        self.best_score_ = self.scores_[best_set_]
        if len(features_to_remove[best_set_]) > 0:
            self.best_features_ = np.delete(
                range(n_features), features_to_remove[best_set_])
        else:
            self.best_features_ = range(n_features)
        self.best_estimator_ = _clone(self.estimator)
        self.best_estimator_.fit(X[:, self.best_features_], y, **fit_params)
        self.n_features_ = len(self.best_features_)
        
        del self.sc
        return self   
    
    def _apply_mask(self, X):
        self._check_is_fitted()
        if self.mask:
            return X[:, self.best_features_]
        else:
            return X
    
    def _check_is_fitted(self):
        _check_is_fitted(self, "best_estimator_")
    
    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def predict(self, X):
        self._check_is_fitted()
        return self.best_estimator_.predict(self._apply_mask(X))

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def predict_proba(self, X):
        self._check_is_fitted()
        return self.best_estimator_.predict_proba(self._apply_mask(X))

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def predict_log_proba(self, X):
        self._check_is_fitted()
        return self.best_estimator_.predict_log_proba(self._apply_mask(X))

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def decision_function(self, X):
        self._check_is_fitted()
        return self.best_estimator_.decision_function(self._apply_mask(X))

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def transform(self, X):
        self._check_is_fitted()
        return self.best_estimator_.transform(self._apply_mask(X))
    
    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def score(self, X, y):
        self._check_is_fitted()
        return self.best_estimator_.score(self._apply_mask(X), y)
        
    @property
    def classes_(self):
        self._check_is_fitted()
        return self.best_estimator_.classes_

