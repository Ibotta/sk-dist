"""
Various postprocessing classes implemented as scikit-learn
transformers compatible with scikit-learn pipelines.
"""

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import Bunch

from .distribute.validation import _check_is_fitted

__all__ = [
    "SimpleVoter"
    ]

class SimpleVoter(BaseEstimator, ClassifierMixin):
    """
    Simple voting classifier which takes fitted estimators as
    input and applies voting ensembling at predict time. Similar to
    `sklearn.enseble.VotingClassifier` except the VotingClassifier
    will actually fit the child estimators, where the SimpleVoter
    will take fitted estimators at instantiation and apply similar
    prediction functions as predict time. 
    
    The VotingClassifier implements a dummy fit method that does nothing
    unless the class attributes `estimators` or `classes` have been 
    manually updated.
    
    The value add over `sklearn.enseble.VotingClassifier` is that
    SimpleVoter can solely run the predict, and let the fit live elsewhere.
    
    Args:
        estimators (list of tuples): list of fitted (name, estimator)
            tuples for voting
        classes (array-like): list of class names from one
            of the underlying estimators
        voting (str): voting method ('hard' or 'soft')
        weights (array-like): array of weights corresponding
            to each estimator
    
    """
    def __init__(self, estimators, classes, voting='hard', weights=None):
        self.estimators = estimators
        self.classes = classes
        self.voting = voting
        self.weights = weights
        self._assemble_attributes()
        
    @property
    def named_estimators(self):
        """ Bunches the estimators by name """
        return Bunch(**dict(self.estimators))
        
    @property
    def _weights_not_none(self):
        """Get the weights of not `None` estimators"""
        if self.weights is None:
            return None
        return [w for est, w in zip(self.estimators, self.weights)
                if est[1] not in (None, 'drop')]
        
    def fit(self, X, y=None):
        """ Trivial fit method; re-assembles attributes """
        self._assemble_attributes()
        return self

    def predict(self, X):
        """ Compute predictions for samples in X """
        _check_is_fitted(self, "estimators_")
        if self.voting == 'soft':
            maj = np.argmax(self.predict_proba(X), axis=1)
        else:  # 'hard' voting
            predictions = self._predict(X)
            maj = np.apply_along_axis(
                lambda x: np.argmax(
                    np.bincount(x, weights=self._weights_not_none)),
                axis=1, arr=predictions)
        maj = self.le_.inverse_transform(maj)
        return maj
        
    @property
    def predict_proba(self):
        """ Compute probabilities of possible outcomes for samples in X """
        return self._predict_proba
        
    def _predict(self, X):
        "" "Collect results from clf.predict calls """
        return np.asarray([self.le_.transform(clf.predict(X)) for clf in self.estimators_]).T
        
    def _predict_proba(self, X):
        """ Predict class probabilities for X in 'soft' voting """
        if self.voting == 'hard':
            raise AttributeError("predict_proba is not available when"
                                 " voting=%r" % self.voting)
        _check_is_fitted(self, "estimators_")
        avg = np.average(self._collect_probas(X), axis=0,
                         weights=self._weights_not_none)
        return avg
        
    def _collect_probas(self, X):
        """Collect results from clf.predict calls. """
        return np.asarray([clf.predict_proba(X) for clf in self.estimators_])
        
    def _assemble_attributes(self):
        """ Assemble fitted class attributes """
        names, clfs = zip(*self.estimators)
        self.estimators_ = clfs
        self.classes_ = self.classes
        self.le_ = LabelEncoder()
        self.le_.classes_ = self.classes
