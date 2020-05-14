"""
Distributed grid search meta-estimators
"""

import time
import numbers
import random
import numpy as np
import pandas as pd

from copy import copy
from abc import ABCMeta
from joblib import Parallel, delayed
from sklearn.model_selection import (
    ParameterGrid, GridSearchCV, 
    RandomizedSearchCV, ParameterSampler, 
    check_cv
    )
from sklearn.metrics import check_scoring
from sklearn.base import BaseEstimator, is_classifier
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import indexable

from functools import partial
from numpy.ma import MaskedArray
from scipy.stats import rankdata
from scipy.sparse import issparse
from itertools import product
from collections import defaultdict

from .validation import (
    _check_estimator, _check_base_estimator, 
    _validate_params, _validate_models, 
    _validate_names, _validate_estimators, 
    _check_n_iter, _is_arraylike, _num_samples,
    _safe_indexing, _check_is_fitted
    )
from .utils import (
    _multimetric_score, _num_samples, 
    _aggregate_score_dicts, _score, 
    _check_multimetric_scoring,
    _safe_split, _dict_slice_remove
    )
from .base import (
    _clone, _get_value, _parse_partitions
    )

__all__ = [
    "DistGridSearchCV", 
    "DistRandomizedSearchCV",
    "DistMultiModelSearch"
]

def _sample_one(n_iter, param_distributions, random_state=None):
    """ Sample from param distributions for one model """
    return list(ParameterSampler(
        param_distributions, 
        n_iter=_check_n_iter(n_iter, param_distributions), 
        random_state=random_state
        ))

def _raw_sampler(models, n_params=None, n=None, 
                 random_state=None):
    """ Sample from param distributions for each model """
    if n_params is None:
        if n is None:
            raise Exception(
                "Must supply either 'n_params' or 'n' as arguments")
        else:
            n_params = [n]*len(models)
    param_sets = []
    for index in range(len(models)):
        sampler = _sample_one(
            n_params[index], models[index][2], 
            random_state=random_state
            )
        for sample_index in range(len(sampler)):
            param_set = {
                "model_index": index, 
                "params_index": sample_index, 
                "param_set": sampler[sample_index]
                }
            param_sets.append(param_set)
    return param_sets

def _fit_one_fold(fit_set, models, X, y, scoring, fit_params):
    """
    Fits the given estimator on one fold of training data.
    Scores the fitted estimator against the test fold.
    """
    train = fit_set[0][0]
    test = fit_set[0][1]
    estimator_ = _clone(models[fit_set[1]["model_index"]][1])
    parameters = fit_set[1]["param_set"]
    X_train, y_train = _safe_split(estimator_, X, y, train)
    X_test, y_test = _safe_split(estimator_, X, y, test, train)
    if parameters is not None:
        estimator_.set_params(**parameters)
    estimator_.fit(X_train, y_train, **fit_params)
    scorer = check_scoring(estimator_, scoring=scoring)
    is_multimetric = not callable(scorer)
    out_dct = fit_set[1]
    out_dct["score"] = _score(
        estimator_, X_test, y_test, 
        scorer, is_multimetric
        )
    return out_dct

def _fit_batch(X, y, folds, param_sets, models, n, 
               scoring, fit_params, random_state=None, 
               sc=None, partitions="auto", n_jobs=None):
    """
    Fits a batch of combinations of parameter sets, models
    and cross validation folds. Returns results pandas
    DataFrames.
    """
    fit_sets = product(folds, param_sets)
    if sc is None:
        scores = Parallel(n_jobs=n_jobs)(
            delayed(_fit_one_fold)(
                x, models, X, y, 
                scoring, fit_params
                )
            for x in fit_sets)
    else:
        fit_sets = list(fit_sets)
        partitions = _parse_partitions(partitions, len(fit_sets))
        scores = (
            sc
            .parallelize(fit_sets, numSlices=partitions)
            .map(lambda x: _fit_one_fold(x, models, X, y, scoring, fit_params))
            .collect()
            )
    param_results = _get_results(scores)
    model_results = (
        param_results
        .groupby(["model_index"])["score"]
        .max()
        .reset_index()
        .sort_values("model_index")
        )
    return param_results, model_results

def _get_results(scores):
    """ Converts 'scores' list to pandas DataFrame """
    cols = [
        "model_index", "params_index", 
        "param_set", "score"
        ]
    df = (
        pd.DataFrame(scores, columns=cols)
        .sort_values(["model_index", "params_index"])
        )
    if len(df) == 0:
        return pd.DataFrame(columns=cols)
    return (
        df
        .groupby(["model_index", "params_index"])
        .agg({"score": "mean", "param_set": "first"})
        .reset_index()
        .sort_values(["model_index", "params_index"])
        [cols]
        )

def _index_param_value(X, v, indices):
    """ Private helper function for parameter value indexing """
    if not _is_arraylike(v) or _num_samples(v) != _num_samples(X):
        return v
    if issparse(v):
        v = v.tocsr()
    return _safe_indexing(v, indices)

def _fit_and_score(estimator, X, y, scorer, train, test, verbose,
                   parameters, fit_params, return_train_score=False,
                   return_parameters=False, return_n_test_samples=False,
                   return_times=False, error_score='raise'):
    """ Fit estimator and compute scores for a given dataset split """
    estimator_ = _clone(_get_value(estimator))
    if verbose > 1:
        if parameters is None:
            msg = ''
        else:
            msg = '%s' % (', '.join('%s=%s' % (k, v)
                          for k, v in list(parameters.items())))
        print(("[CV] %s %s" % (msg, (64 - len(msg)) * '.')))

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = dict([(k, _index_param_value(X, v, train))
                      for k, v in list(fit_params.items())])

    test_scores = {}
    train_scores = {}
    if parameters is not None:
        estimator_.set_params(**parameters)

    start_time = time.time()

    X_train, y_train = _safe_split(estimator_, X, y, train)
    X_test, y_test = _safe_split(estimator_, X, y, test, train)

    is_multimetric = not callable(scorer)
    n_scorers = len(list(scorer.keys())) if is_multimetric else 1

    try:
        if y_train is None:
            estimator_.fit(X_train, **fit_params)
        else:
            estimator_.fit(X_train, y_train, **fit_params)

    except Exception as e:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == 'raise':
            raise
        elif isinstance(error_score, numbers.Number):
            if is_multimetric:
                test_scores = dict(list(zip(list(scorer.keys()),
                                   [error_score, ] * n_scorers)))
                if return_train_score:
                    train_scores = dict(list(zip(list(scorer.keys()),
                                        [error_score, ] * n_scorers)))
            else:
                test_scores = error_score
                if return_train_score:
                    train_scores = error_score
            warnings.warn("Classifier fit failed. The score on this train-test"
                          " partition for these parameters will be set to %f. "
                          "Details: \n%r" % (error_score, e), FitFailedWarning)
        else:
            raise ValueError("error_score must be the string 'raise' or a"
                             " numeric value. (Hint: if using 'raise', please"
                             " make sure that it has been spelled correctly.)")

    else:
        fit_time = time.time() - start_time
        # _score will return dict if is_multimetric is True
        test_scores = _score(estimator_, X_test, y_test, scorer, is_multimetric)
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            train_scores = _score(estimator_, X_train, y_train, scorer,
                                  is_multimetric)

    if verbose > 2:
        if is_multimetric:
            for scorer_name, score in list(test_scores.items()):
                msg += ", %s=%s" % (scorer_name, score)
        else:
            msg += ", score=%s" % test_scores
    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = "%s, total=%s" % (msg, logger.short_format_time(total_time))
        print(("[CV] %s %s" % ((64 - len(end_msg)) * '.', end_msg)))

    ret = [train_scores, test_scores] if return_train_score else [test_scores]

    if return_n_test_samples:
        ret.append(_num_samples(X_test))
    if return_times:
        ret.extend([fit_time, score_time])
    if return_parameters:
        ret.append(parameters)
    return ret
    
class DistBaseSearchCV(BaseEstimator, metaclass=ABCMeta):
    """
    Same as sklearn `BaseSearchCV` but with distributed
    training using spark

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
        preds (bool): keep predictions as attribute
    """
    def __init__(self, estimator, sc=None, partitions='auto', preds=False):
        self.estimator = estimator
        self.sc = sc
        self.partitions = partitions
        self.preds = preds
        
    def fit(self, X, y=None, groups=None, **fit_params):
        """
        Run fit with all sets of parameters. Parallelize fit operations
        using spark.
        
        Args:
            X (array-like, shape = [n_samples, n_features]): training vector, 
                where n_samples is the number of samples and
                n_features is the number of features
            y (array-like, shape = [n_samples] or [n_samples, n_output]): target 
                relative to X for classification or regression
            groups (array-like, with shape (n_samples,)): group labels for 
                the samples used while splitting the dataset into
                train/test set
            **fit_params (dict of string -> object): parameters passed 
                to the ``fit`` method of the estimator
        """
        _check_estimator(self, verbose=self.verbose)
        estimator = self.estimator
        cv = check_cv(self.cv, y, classifier=is_classifier(estimator))

        scorers, self.multimetric_ = _check_multimetric_scoring(
            self.estimator, scoring=self.scoring)

        if self.multimetric_:
            if self.refit is not False and (
                    not isinstance(self.refit, six.string_types) or
                    # This will work for both dict / list (tuple)
                    self.refit not in scorers):
                raise ValueError("For multi-metric scoring, the parameter "
                                 "refit must be set to a scorer key "
                                 "to refit an estimator with the best "
                                 "parameter setting on the whole data and "
                                 "make the best_* attributes "
                                 "available for that metric. If this is not "
                                 "needed, refit should be set to False "
                                 "explicitly. %r was passed." % self.refit)
            else:
                refit_metric = self.refit
        else:
            refit_metric = 'score'

        X, y, groups = indexable(X, y, groups)
        n_splits = cv.get_n_splits(X, y, groups)
        # Regenerate parameter iterable for each fit
        candidate_params = list(self._get_param_iterator())
        n_candidates = len(candidate_params)
        if self.verbose > 0:
            print(("Fitting {0} folds for each of {1} candidates, totalling"
                  " {2} fits".format(n_splits, n_candidates,
                                     n_candidates * n_splits)))

        base_estimator = _clone(self.estimator)
        pre_dispatch = self.pre_dispatch

        fit_sets = []
        cv_splitted = list(cv.split(X, y, groups))
        count = -1
        for fit_set in product(candidate_params, cv_splitted):
            count += 1
            fit_sets.append((count,) + fit_set)
        verbose = self.verbose
        return_train_score = self.return_train_score
        error_score = self.error_score

        if self.sc is None:
            base_estimator_ = base_estimator
            out = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_and_score)(
                    base_estimator_, X, y, scorers, x[2][0], x[2][1],
                    verbose, x[1], fit_params=fit_params,
                    return_train_score=return_train_score,
                    return_n_test_samples=True,
                    return_times=True, return_parameters=False,
                    error_score=error_score)
                for x in fit_sets)
            out = [[fit_sets[ind][0], out[ind]] for ind in range(len(fit_sets))]
        else:
            base_estimator_ = self.sc.broadcast(base_estimator)
            partitions = _parse_partitions(self.partitions, len(fit_sets))
            out = self.sc.parallelize(fit_sets, numSlices=partitions).map(lambda x: [x[0], _fit_and_score(
                base_estimator_, X, y, scorers, x[2][0], x[2][1], 
                verbose, x[1], fit_params=fit_params,
                return_train_score=return_train_score,
                return_n_test_samples=True,
                return_times=True, return_parameters=False,
                error_score=error_score)]).collect()

        out = [out[i][1] for i in np.argsort([x[0] for x in out])]

        # if one choose to see train score, "out" will contain train score info
        if self.return_train_score:
            (train_score_dicts, test_score_dicts, test_sample_counts, fit_time,
             score_time) = list(zip(*out))
        else:
            (test_score_dicts, test_sample_counts, fit_time,
             score_time) = list(zip(*out))

        # test_score_dicts and train_score dicts are lists of dictionaries and
        # we make them into dict of lists
        test_scores = _aggregate_score_dicts(test_score_dicts)
        if self.return_train_score:
            train_scores = _aggregate_score_dicts(train_score_dicts)

        results = {}

        def _store(key_name, array, weights=None, splits=False, rank=False):
            """ A small helper to store the scores/times to the cv_results_ """
            array = np.array(array, dtype=np.float64).reshape(n_candidates,
                                                              n_splits)
            if splits:
                for split_i in range(n_splits):
                    # Uses closure to alter the results
                    results["split%d_%s"
                            % (split_i, key_name)] = array[:, split_i]

            array_means = np.average(array, axis=1, weights=weights)
            results['mean_%s' % key_name] = array_means
            # Weighted std is not directly available in numpy
            array_stds = np.sqrt(np.average((array -
                                             array_means[:, np.newaxis]) ** 2,
                                            axis=1, weights=weights))
            results['std_%s' % key_name] = array_stds

            if rank:
                results["rank_%s" % key_name] = np.asarray(
                    rankdata(-array_means, method='min'), dtype=np.int32)

        _store('fit_time', fit_time)
        _store('score_time', score_time)
        param_results = defaultdict(partial(MaskedArray,
                                            np.empty(n_candidates,),
                                            mask=True,
                                            dtype=object))
        for cand_i, params in enumerate(candidate_params):
            for name, value in list(params.items()):
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurence of `name`.
                # Setting the value at an index also unmasks that index
                param_results["param_%s" % name][cand_i] = value

        results.update(param_results)
        # Store a list of param dicts at the key 'params'
        results['params'] = candidate_params

        # NOTE test_sample counts (weights) remain the same for all candidates
        test_sample_counts = np.array(test_sample_counts[:n_splits],
                                      dtype=np.int)
        for scorer_name in list(scorers.keys()):
            # Computed the (weighted) mean and std for test scores alone
            _store('test_%s' % scorer_name, test_scores[scorer_name],
                   splits=True, rank=True,
                   weights=test_sample_counts if self.iid else None)
            if self.return_train_score:
                prev_keys = set(results.keys())
                _store('train_%s' % scorer_name, train_scores[scorer_name],
                       splits=True)

                if self.return_train_score == 'warn':
                    for key in set(results.keys()) - prev_keys:
                        message = (
                            'You are accessing a training score ({!r}), '
                            'which will not be available by default '
                            'any more in 0.21. If you need training scores, '
                            'please set return_train_score=True').format(key)
                        # warn on key access
                        results.add_warning(key, message, FutureWarning)

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            self.best_index_ = results["rank_test_%s" % refit_metric].argmin()
            self.best_params_ = candidate_params[self.best_index_]
            self.best_score_ = results["mean_test_%s" % refit_metric][
                self.best_index_]

        if self.refit:
            self.best_estimator_ = _clone(base_estimator).set_params(
                **self.best_params_)
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            if self.preds:
                preds = []
                for train_index, test_index in cv_splitted:
                    estimator_ = _clone(base_estimator).set_params(
                        **self.best_params_)
                    estimator_.fit(X[train_index], y[train_index])
                    try:
                        preds.append(estimator_.predict_proba(X[test_index]))
                    except:
                        preds.append(estimator_.predict(X[test_index]))
                self.preds_ = np.vstack(preds)

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers if self.multimetric_ else scorers['score']

        self.cv_results_ = results
        self.n_splits_ = n_splits

        del self.sc
        if hasattr(self.estimator, "sc"):
            del self.estimator.sc
        return self

    def get_preds(self):
        """ Get CV predictions """
        if hasattr(self, "preds_"):
            return self.preds_

    def drop_preds(self):
        """ Remove preds_ attribute """
        if hasattr(self, "preds_"):
            del self.preds_
        
class DistGridSearchCV(DistBaseSearchCV, GridSearchCV):
    """
    Same as sklearn `GridSearchCV` but with distributed
    training using spark.

    Args:
        estimator (estimator object):
            This is assumed to implement the scikit-learn estimator interface.
            Either estimator needs to provide a ``score`` function,
            or ``scoring`` must be passed.
        param_grid (dict or list of dictionaries):
            Dictionary with parameters names (string) as keys and lists of
            parameter settings to try as values, or a list of such
            dictionaries, in which case the grids spanned by each dictionary
            in the list are explored. This enables searching over any sequence
            of parameter settings.
        sc (sparkContext): Spark context for spark broadcasting and rdd operations.
        partitions (int or 'auto'): default 'auto'
            Number of partitions to use for parallelization of parameter
            search space. Integer values or None will be used directly for `numSlices`,
            while 'auto' will set `numSlices` to the number required fits.
        preds (bool): keep predictions as attribute
    """
    def __init__(self, 
                 estimator, 
                 param_grid, 
                 sc=None, 
                 partitions='auto', 
                 preds=False, 
                 scoring=None,
                 n_jobs=None, 
                 iid='warn', 
                 refit=True, 
                 cv=5, 
                 verbose=0,
                 pre_dispatch='2*n_jobs', 
                 error_score='raise-deprecating',
                 return_train_score=False):
        GridSearchCV.__init__(
            self, 
            estimator, 
            param_grid, 
            scoring=scoring,
            n_jobs=n_jobs, 
            iid=iid, 
            refit=refit, 
            cv=cv, 
            verbose=verbose,
            pre_dispatch=pre_dispatch, 
            error_score=error_score,
            return_train_score=return_train_score
            )
        self.sc = sc
        self.partitions = partitions
        self.preds = preds
        
    def _get_param_iterator(self):
        """ Return ParameterGrid instance for the given param_grid """
        return ParameterGrid(self.param_grid)
        
class DistRandomizedSearchCV(DistBaseSearchCV, RandomizedSearchCV):
    """
    Same as sklearn `RandomizedSearchCV` but with distributed
    training using spark.

    Args:
        estimator (estimator object):
            This is assumed to implement the scikit-learn estimator interface.
            Either estimator needs to provide a ``score`` function,
            or ``scoring`` must be passed.
        param_distributions (dict):
            Dictionary with parameters names (string) as keys and distributions
            or lists of parameters to try. Distributions must provide a ``rvs``
            method for sampling (such as those from scipy.stats.distributions).
            If a list is given, it is sampled uniformly.
        sc (sparkContext): Spark context for spark broadcasting and rdd operations.
        partitions (int or 'auto'): default 'auto'
            Number of partitions to use for parallelization of parameter
            search space. Integer values or None will be used directly for `numSlices`,
            while 'auto' will set `numSlices` to the number required fits.
        preds (bool): keep predictions as attribute
    """
    def __init__(self, 
                 estimator, 
                 param_distributions, 
                 sc=None, 
                 partitions='auto', 
                 preds=False, 
                 n_iter=10, 
                 scoring=None,
                 n_jobs=None, 
                 iid='warn', 
                 refit=True,
                 cv=5, 
                 verbose=0, 
                 pre_dispatch='2*n_jobs',
                 random_state=None, 
                 error_score='raise-deprecating',
                 return_train_score=False):
        RandomizedSearchCV.__init__(
            self, 
            estimator, 
            param_distributions, 
            n_iter=n_iter, 
            scoring=scoring,
            n_jobs=n_jobs, 
            iid=iid, 
            refit=refit,
            cv=cv, 
            verbose=verbose, 
            pre_dispatch=pre_dispatch,
            random_state=random_state, 
            error_score=error_score,
            return_train_score=return_train_score
            )
        self.sc = sc
        self.partitions = partitions
        self.preds = preds
        
    def _get_param_iterator(self):
        """ Return ParameterSampler instance for the given distributions """
        return ParameterSampler(
            self.param_distributions, self.n_iter,
            random_state=self.random_state)

class DistMultiModelSearch(BaseEstimator, metaclass=ABCMeta):
    """
    Distributed multi-model search meta-estimator. Similar to 
    `DistRandomizedSearchCV` but with handling for multiple models.
    Takes a `models` input containing a list of tuples, each with a 
    string name, instantiated estimator object, and parameter set
    dictionary for random search.

    The fit method will compute a cross validation score for each
    estimator/parameter set combination after randomly sampling from
    the parameter set for each estimator. The best estimator/parameter
    set combination will be refit if appropriate. The process is distrubuted
    with spark if a sparkContext is provided, else joblib is used.

    Args:
        models (array-like): List of tuples containing estimator and parameter
            set information to generate candidates. Each tuple is of the form:
            ('name' <str>, 'estimator' <sklearn Estimator>, 'param_set' <dict>)
            For example: ('rf', RandomForestClassifier(), {'max_depth': [5,10]})
        sc (sparkContext): Spark context for spark broadcasting and rdd operations.
        partitions (int or 'auto'): default 'auto'
            Number of partitions to use for parallelization of parameter
            search space. Integer values or None will be used directly for `numSlices`,
            while 'auto' will set `numSlices` to the number required fits.
        n (int): Number of parameter sets to sample from parameter space for each
            estimator.
        cv (int, cross-validation generator or an iterable): Determines the 
            cross-validation splitting strategy.
        scoring (string, callable, list/tuple, dict or None): A single string or a 
            callable to evaluate the predictions on the test set. If None, 
            the estimator's score method is used.
        random_state (int): Random state used throughout to ensure consistent runs.
        verbose (int, bool): Used to indicate level of stdout logging. 
        refit (bool): Refits best estimator at the end of fit method.
        n_jobs (int): Number of jobs to run in parallel. Only used if sc=None.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. 
        pre_dispatch (int): Controls the number of jobs that get dispatched 
            during parallel execution. Reducing this number can be useful 
            to avoid an explosion of memory consumption when more jobs 
            get dispatched than CPUs can process. Only used if sc=None. 
    """
    def __init__(self, 
                 models,
                 sc=None,
                 partitions="auto",
                 n=5,
                 cv=5,
                 scoring=None,
                 random_state=None,
                 verbose=0, 
                 refit=True, 
                 n_jobs=None, 
                 pre_dispatch='2*n_jobs'):
        self.models = models
        self.sc = sc
        self.partitions = partitions
        self.n = n
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.verbose = verbose
        self.refit = refit 
        self.n_jobs = n_jobs 
        self.pre_dispatch = pre_dispatch
        
    def fit(self, X, y=None, groups=None, **fit_params):
        """
        Run fit with all sets of parameters. Parallelize fit operations
        using spark.
        
        Args:
            X (array-like, shape = [n_samples, n_features]): training vector, 
                where n_samples is the number of samples and
                n_features is the number of features
            y (array-like, shape = [n_samples] or [n_samples, n_output]): target 
                relative to X for classification or regression
            groups (array-like, with shape (n_samples,)): group labels for 
                the samples used while splitting the dataset into
                train/test set
            **fit_params (dict of string -> object): parameters passed 
                to the ``fit`` method of the estimator
        """
        _check_estimator(self, verbose=self.verbose)
        models = _validate_models(self.models, self)
        cv = check_cv(
            self.cv, y, classifier=is_classifier(models[0][1]))
        folds = list(cv.split(X,y,groups))
        results = pd.DataFrame()
        
        def _sample_generator(models, n, random_state): 
            rs = None if random_state is None else random_state*(i+1)
            yield _raw_sampler(models, n=n, random_state=random_state)
                
        sample_gen = _sample_generator(
            models, n=self.n, random_state=self.random_state)
        param_sets = list(sample_gen)[0]
        results, model_results = _fit_batch(
            X, y, folds, param_sets, models, self.n, 
            self.scoring, fit_params, sc=self.sc, 
            n_jobs=self.n_jobs,
            partitions=self.partitions,
            random_state=self.random_state, 
            )
        if self.verbose:
            print(model_results)
        
        best_index = np.argmax(results["score"].values)
        self.best_model_index_ = results.iloc[best_index]["model_index"]
        self.best_model_name_ = models[self.best_model_index_][0]
        self.best_params_ = results.iloc[best_index]["param_set"]
        self.best_score_ = results.iloc[best_index]["score"]
        self.worst_score_ = results.iloc[best_index]["score"]
        
        results["rank_test_score"] = np.asarray(
            rankdata(-results["score"].values), 
            dtype=np.int32
            )
        results["mean_test_score"] = results["score"]
        results["params"] = results["param_set"]
        results["model_name"] = (
            results["model_index"]
            .apply(lambda x: models[x][0])
            )
        result_cols = [
            "model_index", "model_name", "params", 
            "rank_test_score", "mean_test_score"
            ]
        self.cv_results_ = results[result_cols].to_dict(orient="list")
        
        if self.refit:
            self.best_estimator_ = _clone(models[self.best_model_index_][1])
            self.best_estimator_.set_params(**self.best_params_)
            self.best_estimator_.fit(X, y, **fit_params)
        
        del self.sc
        return self
    
    def _check_is_fitted(self):
        if not self.refit:
            raise NotFittedError('This %s instance was initialized '
                                 'with refit=False. The method is '
                                 'available only after refitting on the best '
                                 'parameters. You can refit an estimator '
                                 'manually using the ``best_params_`` '
                                 'attribute'
                                 % (type(self).__name__))
        else:
            _check_is_fitted(self, "best_estimator_")
    
    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def predict(self, X):
        self._check_is_fitted()
        return self.best_estimator_.predict(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def predict_proba(self, X):
        self._check_is_fitted()
        return self.best_estimator_.predict_proba(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def predict_log_proba(self, X):
        self._check_is_fitted()
        return self.best_estimator_.predict_log_proba(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def decision_function(self, X):
        self._check_is_fitted()
        return self.best_estimator_.decision_function(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def transform(self, X):
        self._check_is_fitted()
        return self.best_estimator_.transform(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def inverse_transform(self, Xt):
        self._check_is_fitted()
        return self.best_estimator_.inverse_transform(Xt)

    @property
    def classes_(self):
        self._check_is_fitted()
        return self.best_estimator_.classes_
