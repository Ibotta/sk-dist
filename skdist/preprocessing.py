"""
Various preprocessing classes implemented as scikit-learn
transformers compatible with scikit-learn pipelines.
"""

import pandas as pd
import numpy as np
import numbers

from past.builtins import basestring
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn import feature_selection

__all__ = [
    VarSelect, 
    CastToType, 
    FillNA, 
    DenseTransformer, 
    LabelEncoderPipe, 
    SelectorMem, 
    HashingVectorizerChunked
    ]

_SELECTOR_LOOKUP = {
    "fpr": feature_selection.SelectFpr,
    "fdr": feature_selection.SelectFdr,
    "kbest": feature_selection.SelectKBest,
    "percentile": feature_selection.SelectPercentile,
    "fwe": feature_selection.SelectFwe
}

class VarSelect(BaseEstimator, TransformerMixin):
    """
    Select a set of columns from a numpy array or pandas dataframe. 
    """
    def __init__(self, vars_to_select, cast_to_numpy=True, flatten=True):
        """
        Constructor.

        Args:
            vars_to_select (list of int/strings or string): If integers then the indices
              of columns to select and if strings then the panda dataframe
              column names to select.
            cast_to_numpy (Boolean): Should pandas DF be convereted to a numpy
              array?
            flatten (Boolean): Should X be flattened before returned? Will only flatten if 1-d.
        """
        self.vars_to_select = vars_to_select
        self.cast_to_numpy = cast_to_numpy
        self.flatten = flatten

    def fit(self, X, y=None):
        """
        Trivial fit method.

        Args:
            X: Training set.
            y: Target.
        """
        return self

    def transform(self, X):
        """
        Select columns from numpy array/pandas dataframe.

        Args:
            X (numpy array or pandas dataframe): Data set.

        Returns:
            Data set with desired columns.
        """
        self._process_args()
        if not hasattr(self, 'flatten'):
            self.flatten = True

        if isinstance(X, pd.DataFrame):
            X = X[self.varnames]
            if self.cast_to_numpy:
                X = X.values
        else:
            X = X[:,self.varindex]

        if self.flatten:
            if X.shape[1] == 1 and isinstance(X, np.ndarray):
                X = X.flatten()
        return X

    def _process_args(self):
        """ Process input arguments """
        if hasattr(self, 'vars_to_select') and self.vars_to_select is not None:
            if isinstance(self.vars_to_select, basestring):
                self.vars_to_select = [self.vars_to_select]
            if all(isinstance(x, numbers.Integral) for x in self.vars_to_select):
                self.varindex = self.vars_to_select
            elif all(isinstance(x, basestring) for x in self.vars_to_select):
                self.varnames = self.vars_to_select

class CastToType(BaseEstimator, TransformerMixin):
    """
    Cast a numpy array or pandas dataframe to a numpy array of floats.
    """
    def __init__(self, type):
        self.type=type

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            return X.values.astype(self.type)
        else:
            return X.astype(self.type)

class FillNA(BaseEstimator, TransformerMixin):
    """
    Fill missing values from a numpy array or pandas dataframe. 
    """
    def __init__(self, fill_value, vars_to_fillna=None, cast_to_numpy=True):
        """
        Constructor.

        Args:
            fill_value (string/int/float): Value to fill in for missing values
            vars_to_select (list of int/strings or string): If None then fill 
                all NA. Else, only fill in for the selected columns
            cast_to_numpy (Boolean): Should pandas DF be convereted to a numpy
              array?
        """
        if vars_to_fillna:
            if isinstance(vars_to_fillna, basestring):
                vars_to_fillna = [vars_to_fillna]
            elif all(isinstance(x, numbers.Integral) for x in vars_to_fillna):
                self.varindex = vars_to_fillna

        self.vars_to_fillna = vars_to_fillna
        self.fill_value = fill_value
        self.cast_to_numpy = cast_to_numpy

    def fit(self, X, y=None):
        """
        Trivial fit method.

        Args:
            X: Training set.
            y: Target.
        """
        return self

    def transform(self, X):
        """
        Select columns from numpy array/pandas dataframe.

        Args:
            X (numpy array or pandas dataframe): Data set.

        Returns:
            Data set with desired null columns filled.
        """

        if self.vars_to_fillna:
            if isinstance(X, pd.DataFrame):
                X[self.vars_to_fillna] = X[self.vars_to_fillna].fillna(self.fill_value)
                if self.cast_to_numpy:
                    X = X.values
            else:
                mask = pd.isnull(X)
                mask_cols = np.zeros_like(X, dtype=bool)
                mask_cols[:,self.varindex] = True
                X[mask_cols & mask] = self.fill_value
        else:
            if isinstance(X, pd.DataFrame):
                X = X.fillna(self.fill_value)
                if self.cast_to_numpy:
                    X = X.values
            else:
                mask = pd.isnull(X)
                X[mask] = self.fill_value
        return X

class DenseTransformer(BaseEstimator, TransformerMixin):
    """
    Transform sparse matrix to dense

    Args:
        X (scipy sparse matrix)

    Returns:
        X np.array
    """
    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit(self, X, y=None, **fit_params):
        return self

class LabelEncoderPipe(TransformerMixin, BaseEstimator):
    """ LabelEncoder wrapper with TransformerMixin """
    def fit(self, X, y=None):
        self.le = LabelEncoder()
        self.le.fit(X)
        return self
        
    def transform(self, X):
        return [[x] for x in self.le.transform(X)]

class SelectorMem(TransformerMixin, BaseEstimator):
    """ 
    Memory efficient feature selector.
    Parameters
    ----------
    selector : object 
        Any sklearn.feature_selection.univariate_selection._BaseFilter
        inheritor.
    """
    def __init__(self, selector="fpr", score_func=feature_selection.f_classif, threshold=0.05):
        self.selector = selector
        self.score_func = score_func
        self.threshold = threshold

    def fit(self, X, y):
        """
        Run score function on (X, y) and get the appropriate features.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """
        selector = _SELECTOR_LOOKUP[self.selector.lower()](self.score_func, self.threshold)
        selector.fit(X, y)
        mask_indices = selector.get_support(indices=True)
        mask_bool = selector.get_support(indices=False)
        if np.array(mask_bool).nbytes > np.array(mask_indices).nbytes:
            self.mask = mask_indices
        else:
            self.mask = mask_bool
        return self

    def transform(self, X):
        """
        Reduce X to the selected features.
        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.
        Returns
        -------
        X_r : array of shape [n_samples, n_selected_features]
            The input samples with only the selected features.
        """
        return X[:, self.mask]

class HashingVectorizerChunked(HashingVectorizer):
    """ """
    def __init__(self, chunksize=100000, **kwargs):
        self.chunksize = chunksize
        HashingVectorizer.__init__(
            self, **kwargs)

    def transform(self, X):
        """Transform a sequence of documents to a document-term matrix.
        Parameters
        ----------
        X : iterable over raw text documents, length = n_samples
            Samples. Each sample must be a text document (either bytes or
            unicode strings, file name or file object depending on the
            constructor argument) which will be tokenized and hashed.
        Returns
        -------
        X : scipy.sparse matrix, shape = (n_samples, self.n_features)
            Document-term matrix.
        """
        if (len(X) < self.chunksize) or (self.chunksize is None):
            return self._transform(X)
        else:
            return sparse.vstack([self._transform(x) for x in self._chunks(X)])

    def _transform(self, X):
        if isinstance(X, six.string_types):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "string object received.")

        analyzer = self.build_analyzer()
        X = self._get_hasher().transform(analyzer(doc) for doc in X)
        if self.binary:
            X.data.fill(1)
        if self.norm is not None:
            X = normalize(X, norm=self.norm, copy=False)
        return X

    def _chunks(self, l):
        for i in range(0, len(l), self.chunksize):
            yield l[i:i + self.chunksize]
