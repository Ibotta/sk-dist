"""
Feature selection models
"""

import numpy as np

from sklearn.base import (
    BaseEstimator, ClassifierMixin, 
    clone, is_classifier
    )
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import check_is_fitted 
from sklearn.utils import check_X_y
from sklearn.model_selection import check_cv
from sklearn.metrics.scorer import check_scoring
from joblib import Parallel, delayed
from itertools import product
from scipy.sparse import issparse

from .base import _parse_partitions, _clone, _safe_split

def _drop_col(X, index):
    """ Drop index columns from numpy array or sparse matrix """
    if issparse(X):
        cols = np.arange(X.shape[1])
        cols_to_keep = np.where(np.logical_not(np.in1d(cols, [5])))[0]
        return X[:, cols_to_keep]
    else:
        return np.delete(X, index, axis=1)

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

class FeatureEliminator(BaseEstimator, ClassifierMixin):
    """
    Apply a leave-one-out feature elimination scheme, fitting the
    base estimator on the input training data once for each left
    out features. Uses cross validation to score each feature.
    This results in `n_features * cv` total fits. Fit sets
    can be distributed with Spark or joblib.
    
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
        n_features (int): Number of features to keep after elimination.
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
                 n_features=None, 
                 cv=5, 
                 scoring=None, 
                 verbose=False, 
                 n_jobs=None, 
                 pre_dispatch=None, 
                 mask=True):
        self.estimator = estimator
        self.sc = sc
        self.partitions = partitions
        self.n_features = n_features
        self.cv = cv
        self.scoring = scoring
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.mask = mask
        
    def fit(self, X, y=None, groups=None, **fit_params):
        X, y = check_X_y(X, y, "csr", ensure_min_features=2)
        cv = check_cv(self.cv, y, is_classifier(self.estimator))
        scorer = check_scoring(self.estimator, scoring=self.scoring)
        n_features = X.shape[1]
        cv_splits_ = list(cv.split(X,y,groups))
        fit_sets = list(product(range(n_features), cv_splits_))
        base_estimator = _clone(self.estimator)
        
        if not self.sc:
            parallel = Parallel(
                n_jobs=self.n_jobs, verbose=self.verbose, 
                pre_dispatch=self.pre_dispatch
                )
            scores = parallel(
                delayed(_fit_and_score_one)(
                    index, clone(base_estimator), X, y, 
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
                    x[0], clone(base_estimator), X, y, scorer, 
                    x[1][0], x[1][1], verbose, fit_params)]).collect()
                )
            score_sets = []
            for index in range(n_features):
                this_set = []
                for row in scores:
                    if row[0] == index:
                        this_set.append(row[1])
                score_sets.append(this_set)
            
        self.scores_ = []
        for score_set in score_sets:
            this_score = np.mean(score_set)
            self.scores_.append(this_score)
            
        if self.n_features is None:
            n_features_ = int(round(n_features / 2.0))
        else:
            n_features_ = self.n_features
        self.best_features_ = np.argsort(self.scores_)[:n_features_]
        
        self.best_estimator_ = clone(self.estimator)
        self.best_estimator_.fit(X[:, self.best_features_],y,**fit_params)
        
        del self.sc
        return self   
    
    def _apply_mask(self, X):
        self._check_is_fitted('best_features_')
        if self.mask:
            return X[:, self.best_features_]
        else:
            return X
    
    def _check_is_fitted(self, method_name):
        check_is_fitted(self, method_name)
    
    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def predict(self, X):
        self._check_is_fitted('predict')
        return self.best_estimator_.predict(self._apply_mask(X))

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def predict_proba(self, X):
        self._check_is_fitted('predict_proba')
        return self.best_estimator_.predict_proba(self._apply_mask(X))

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def predict_log_proba(self, X):
        self._check_is_fitted('predict_log_proba')
        return self.best_estimator_.predict_log_proba(self._apply_mask(X))

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def decision_function(self, X):
        self._check_is_fitted('decision_function')
        return self.best_estimator_.decision_function(self._apply_mask(X))

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def transform(self, X):
        self._check_is_fitted('transform')
        return self.best_estimator_.transform(self._apply_mask(X))
    
    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def score(self, X, y):
        self._check_is_fitted('score')
        return self.best_estimator_.score(self._apply_mask(X), y)
        
    @property
    def classes_(self):
        self._check_is_fitted("classes_")
        return self.best_estimator_.classes_

