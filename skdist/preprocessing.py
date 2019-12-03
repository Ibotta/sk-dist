"""
Various preprocessing classes implemented as scikit-learn
transformers compatible with scikit-learn pipelines.
"""

import warnings
import pandas as pd
import numpy as np

from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    MultiLabelBinarizer, FunctionTransformer, 
    LabelEncoder, normalize
    )
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn import feature_selection

__all__ = [
    "SelectField", 
    "FeatureCast", 
    "ImputeNull", 
    "DenseTransformer", 
    "SparseTransformer", 
    "LabelEncoderPipe", 
    "SelectorMem", 
    "HashingVectorizerChunked",
    "MultihotEncoder"
    ]

_SELECTOR_LOOKUP = {
    "fpr": feature_selection.SelectFpr,
    "fdr": feature_selection.SelectFdr,
    "kbest": feature_selection.SelectKBest,
    "percentile": feature_selection.SelectPercentile,
    "fwe": feature_selection.SelectFwe
}

class _CustomFunctionTransformer(TransformerMixin, BaseEstimator):
    """ Base class for function transformers """
    def fit(self, X, y=None):
        self.transformer_.fit(X, y=None)
        return self

    def transform(self, X, y=None):
        return self.transformer_.transform(X)

def select_field(X, cols=None, single_dimension=False):
    """ 
    Select columns from a pandas DataFrame

    Args:
        X (pandas DataFrame): input data
        cols (array-like): list of columns to select
        single_dimension (bool): reduce data to 
            one dimension if only one column
            is requested
    Returns:
        X (numpy array)
    """
    if cols is None:
        return X.values
    if len(cols) > 1:
        return X[cols].values
    if len(cols) == 1:
        if single_dimension:
            return X[cols[0]].values
        else:
            return X[cols].values

class SelectField(_CustomFunctionTransformer):
    """ 
    Applies `select_field` as FunctionTransformer

    Args:
        cols (array-like): list of columns to select
        single_dimension (bool): reduce data to 
            one dimension if only one column
            is requested
    """
    def __init__(self, cols=None, single_dimension=False):
        self.cols = cols
        self.single_dimension = single_dimension
        kw_args = {"cols": cols, "single_dimension": single_dimension}
        self.transformer_ = FunctionTransformer(
            select_field, 
            kw_args=kw_args, 
            validate=False
            )

def to_dense(X):
    """ Densify sparse matrix if issparse """
    if sparse.issparse(X):
        return X.todense()
    else:
        return X

class DenseTransformer(_CustomFunctionTransformer):
    """ Applies `to_dense` as a FunctionTransformer """
    def __init__(self):
        self.transformer_ = FunctionTransformer(to_dense, validate=False)

def to_sparse(X):
    """ Sparsify dense matrix if not issparse """
    if sparse.issparse(X):
        return X
    else:
        return sparse.csr_matrix(X)

class SparseTransformer(_CustomFunctionTransformer):
    """ Applies `to_sparse` as a FunctionTransformer """
    def __init__(self):
        self.transformer_ = FunctionTransformer(to_sparse, validate=False)

def feature_cast(X, cast_type=None):
    """
    Casts feature data to requeted type

    Args:
        X (numpy array): input data
        cast_type (type): type to cast data
    Returns:
        X (numpy array)
    """
    if cast_type is None:
        return X
    else:
        return X.astype(cast_type)

class FeatureCast(_CustomFunctionTransformer):
    """
    Applies `feature_cast` as a FunctionTransformer

    Args:
        cast_type (type): type to cast data
    """
    def __init__(self, cast_type=None):
        self.transformer_ = FunctionTransformer(
            feature_cast, kw_args={"cast_type": cast_type}, 
            validate=False
            )

def impute_null(X, impute_val=None):
    """
    Impute null values. Null values are 
    determined using `pd.isnull`.

    Args:
        X (numpy array): input data
        impute_val (object): value to impute with
    Returns:
        X (numpy array)
    """
    if impute_val is None:
        return X
    else:
        X[pd.isnull(X)] = impute_val
        return X

class ImputeNull(_CustomFunctionTransformer):
    """
    Applies `impute_null` as a FunctionTransformer

    Args:
        impute_val (object): value to impute with
    """
    def __init__(self, impute_val=None):
        self.transformer_ = FunctionTransformer(
            impute_null, kw_args={"impute_val": impute_val},
            validate=False
            )

class LabelEncoderPipe(TransformerMixin, BaseEstimator):
    """ 
    LabelEncoder wrapper with TransformerMixin. Allows LabelEncoder 
    to work with a Pipeline. 
    """
    def fit(self, X, y=None):
        """ Fit the LabelEncoder """
        self.le = LabelEncoder()
        self.le.fit(X)
        return self
        
    def transform(self, X, y=None):
        """ Transform the label encoder """
        return np.array([[x] for x in self.le.transform(X)])

class SelectorMem(TransformerMixin, BaseEstimator):
    """ 
    Memory efficient feature selector. Identical functionality
    to classes in sklearn.feature_selection.univariate_selection but
    with more memory efficient attribute storage.
    
    Args:
        selector (object): any sklearn.feature_selection.univariate_selection._BaseFilter
            inheritor
        score_func (function): selector scoring function
        threshold (int/float): scoring threshold to use with scoring function and selector
    """
    def __init__(self, selector="fpr", score_func=feature_selection.f_classif, threshold=0.05):
        self.selector = selector
        self.score_func = score_func
        self.threshold = threshold

    def fit(self, X, y=None):
        """
        Run score function on (X, y) and get the appropriate features
        
        Args:
            X (array-like, shape = [n_samples, n_features])
                The training input samples
            y (array-like, shape = [n_samples])
                The target values (class labels in classification, real numbers in
                regression)
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

    def transform(self, X, y=None):
        """
        Reduce X to the selected features

        Args:
            X (array of shape [n_samples, n_features])
                The input samples

        Returns:
            X_r (array of shape [n_samples, n_selected_features])
                The input samples with only the selected features
        """
        return X[:, self.mask]

class HashingVectorizerChunked(HashingVectorizer):
    """
    Equivalent to HashingVectorizer but with chunked prediction

    Args:
        chunksize (int): size of transform chunks
    """
    def __init__(self, chunksize=100000, **kwargs):
        self.chunksize = chunksize
        HashingVectorizer.__init__(
            self, **kwargs)

    def transform(self, X):
        """Transform a sequence of documents to a document-term matrix

        Args:
            X (iterable over raw text documents, length = n_samples)
                Samples. Each sample must be a text document (either bytes or
                unicode strings, file name or file object depending on the
                constructor argument) which will be tokenized and hashed.

        Returns:
            X (scipy.sparse matrix, shape = (n_samples, self.n_features))
                Document-term matrix.
        """
        if (len(X) < self.chunksize) or (self.chunksize is None):
            return self._transform(X)
        else:
            return sparse.vstack([self._transform(x) for x in self._chunks(X)])

    def _transform(self, X):
        if isinstance(X, str):
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

class MultihotEncoder(BaseEstimator, TransformerMixin):
    """
    Wraps `MultiLabelBinarizer` in a pipeline safe transformer

    Args:
        sparse_output (bool): convert output to sparse matrix
    """
    def __init__(self, sparse_output=False):
        self.transformer = MultiLabelBinarizer()
        self.sparse_output = sparse_output

    def fit(self, X, y=None):
        """ Fit MultiLabelBinarizer """
        self.transformer.fit(X)
        return self

    def transform(self, X,y=None):
        """ Transform MultiLabelBinarizer """
        # ignore unseen label warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_t = self.transformer.transform(X)
        if self.sparse_output:
            return sparse.csr_matrix(X_t)
        else:
            return X_t
