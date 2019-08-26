"""
Distributed grid search meta-estimators
"""

import time
import numbers
import numpy as np

from joblib import Parallel, delayed
from sklearn.model_selection._search import (
    ParameterGrid, BaseSearchCV, 
    GridSearchCV, RandomizedSearchCV,
    ParameterSampler
    )
from sklearn.model_selection._split import check_cv
from sklearn.base import is_classifier
from sklearn.metrics.scorer import check_scoring
from sklearn.utils.validation import indexable
from sklearn.utils.fixes import MaskedArray

from functools import partial
from scipy.stats import rankdata
from itertools import product
from collections import defaultdict

from .base import (
    _check_estimator, _safe_split, _clone,
    _get_value, _parse_partitions
    )

__all__ = [
    "DistGridSearchCV", 
    "DistRandomizedSearchCV"
]

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
    
class DistBaseSearchCV(BaseSearchCV):
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
        **kwargs: Keyword arguments to be passed to `BaseSearchCV`.
    """
    def __init__(self, estimator, sc=None, partitions='auto', preds=False, **kwargs):
        BaseSearchCV.__init__(
            self, estimator, **kwargs)
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
