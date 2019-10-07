"""
Distributed, flexible feature encoder
"""

import ast
import numpy as np

from pandas import DataFrame
from scipy import sparse
from copy import copy
from joblib import Parallel, delayed

from sklearn.pipeline import FeatureUnion
from sklearn.base import TransformerMixin, BaseEstimator

from .base import _parse_partitions 

__all__ = [
    "Encoderizer",
    "EncoderizerExtractor"
]

def _transform_one(transformer, weight, X):
    res = transformer.transform(X)
    # if we have a weight for this transformer, multiply output
    if weight is None:
        return res
    return res * weight

def _fit_one_transformer(transformer, X, y):
    return transformer.fit(X, y)

class Encoderizer(FeatureUnion):
    """ 
    Generic feature encoder with flexible data type input, 
    optional bottom up encoder type inference, and 
    top down `FeatureUnion` configuration acceptance. Also
    includes feature origin method to track down the origin
    transformer of a given feature. Can optionally distribute
    feature union fitting using spark.
    
    Includes `extract` method for slicing off peices
    of the fitted `FeatureUnion` pipeline as a copy of 
    itself with only the requested steps.

    Args:
        transformer_list (iterable or array-like):
            List of transformers, similar to that of FeatureUnion
        transformer_wieghts (dict):
            Multiplicative weights for features per transformer. 
            Keys are transformer names, values the weights.
        n_jobs (int):
            Number of jobs for joblib parallelization
        size (string, 'small', 'medium', or 'large')
            Size of default encoder if using encoder inference
        col_names (iterable or array-like):
            List of column names if using numpy input
        config (dict):
            Configuration of column data types. Keys are column names
            and values are names of default encoder types.
        sc (sparkContext): Spark context for spark broadcasting and rdd operations.
        partitions (int or 'auto'): default 'auto'
            Number of partitions to use for parallelization of parameter
            search space. Integer values or None will be used directly for `numSlices`,
            while 'auto' will set `numSlices` to the number required fits.
    """
    def __init__(self, 
            transformer_list=None, transformer_weights=None,
            n_jobs=1, size="small", config=None,
            col_names=None, sc=None, partitions='auto'):
        self.transformer_list = transformer_list
        self.transformer_weights = transformer_weights
        self.n_jobs = n_jobs
        self.size = size
        self.config = config
        self.col_names = col_names
        self.sc = sc
        self.partitions = partitions
            
    def extract(self, step_names):
        """ 
        Extract copy of fitted self with slice of transformer list
        
        Args:
            step_names (list): White list of transformer
            names to include in extraction slice
            
        Returns:
            fitted Encoderizer object
        """
        encoderizer = copy(self)
        encoderizer.transformer_lengths = [
            encoderizer.transformer_lengths[x] 
            for x in range(len(encoderizer.step_names))
            if encoderizer.step_names[x] in step_names
            ]
        encoderizer.transformer_list = [
            encoderizer.transformer_list[x]
            for x in range(len(encoderizer.step_names))
            if encoderizer.step_names[x] in step_names
            ]        
        return encoderizer
    
    def fit(self, X, y=None):
        """
        Fit all transformers using X

        Args:
            X (iterable or array-like):
                Input data, used to fit transformers.
            y (array-like, shape (n_samples, ...), optional):
               Targets for supervised learning.
        Returns:
            self (Encoderizer object)
                This estimator
        """
        X = self._process_input(X)
        if self.transformer_list is None:
            self.transformer_list = self._infer_transformers(X)
        self.transformer_list = list(self.transformer_list)
        self._validate_transformers()
        if self.sc is None:
            transformers = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_one_transformer)(trans, X, y)
                for _, trans, _ in self._iter())
        else:
            partitions = _parse_partitions(self.partitions, len(self.transformer_list))
            trans_rdd = self.sc.parallelize(
                list(zip(list(range(len(list(self._iter())))), [x[1] for x in self._iter()])), 
                numSlices=partitions)
            def fot(trans, X, y):
                if y is not None:
                    return _fit_one_transformer(trans, X, y)
                else:
                    return _fit_one_transformer(trans, X, None)
            transformers = trans_rdd.map(lambda x: [x[0], fot(x[1], X, y)]).collect()
            indices = [x[0] for x in transformers]
            trans_objs = [x[1] for x in transformers]
            transformers = list([trans_objs[indices.index(x)] for x in indices])
        self._update_transformer_list(transformers)
        self._feature_indices(X)
        del self.sc
        return self
        
    def transform(self, X):
        """
        Transform X separately by each transformer, concatenate results

        Args:
            X (iterable or array-like):
                Input data to be transformed.
        Returns:
            X_t (array-like or sparse matrix):
                hstack of results of transformers. sum_n_components is the
                sum of n_components (output dimension) over transformers.
        """
        X = self._process_input(X, fit=False)
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, weight, X)
            for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.hstack(Xs)
        return Xs

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit to data, then transform it. Fits transformer to 
        X and y with optional parameters fit_params and returns 
        a transformed version of X.
        
        Args:
            X (numpy array of shape [n_samples, n_features])
                Training set.
            y (numpy array of shape [n_samples]):
                Target values.
   
        Returns:
            X_new (numpy array of shape [n_samples, n_features_new])
                Transformed array.
        """
        if y is None:
            return self.fit(X, **fit_params).transform(X)
        else:
            return self.fit(X, y, **fit_params).transform(X)

    def feature_origin(self, index, mask=None):
        """
        Return step name given a feature index. Optionally 
        supply a custom mask if used in a supplemental feature selection
        step late in the pipeline.

        Args:
            index (int):
                Index of desired feature in transformed
                feature vector.
            mask (numpy ndarray, default None):
                optional boolean mask from feature selector used
                later in the transformation pipeline.
        Returns:
            step_name (string): 
                Name of feature union step of origin
                transformer.
        """
        cumulative = np.cumsum(self.transformer_lengths)
        if mask is not None:
            cumulative = np.array([
                (mask[x - 1]) 
                for x in cumulative])
        return self.step_names[np.argmax(cumulative > index)]
        
    @property
    def step_names(self):
        """ Get ordered list of step names """
        return [x[0] for x in self.transformer_list]
        
    def _process_input(self, X, fit=True):
        """ 
        Converts flexible intput type into pandas DataFrame. Handles
        pandas DataFrame, dictionary, pyspark DataFrame, or 
        numpy ndarray.
        """
        if isinstance(X, DataFrame):
            out = X
        elif isinstance(X, dict):
            try:
                out = DataFrame.from_dict(
                    X, orient="columns")
            except:
                raise ValueError("Cannot parse input")
        elif isinstance(X, np.ndarray) or isinstance(X, list):
            if fit and self.col_names is None:
                raise ValueError(
                    "Must supply col_names with numpy array input")
            elif fit:
                out = DataFrame(X, columns=self.col_names)
            else:
                out = DataFrame(X, columns=self.fields_)
        else:
            from pyspark.sql import DataFrame as SparkDataFrame
            if isinstance(X, SparkDataFrame):
                out = X.toPandas()
            else:
                raise ValueError("Cannot parse input")
        if fit:
            self.fields_ = list(out.columns)
        return out
            
    def _infer_transformers(self, X):
        """ 
        Infer transformer steps based on data types
        and distributions. Use config instead if given.
        """
        from ._defaults import _default_encoders
        if self.config is not None:
            lst = [
                _default_encoders[self.size][v](c)
                for c, v in self.config.items()]
        else:
            lst = [
                self._infer_column(
                    c, X[c], 
                    _default_encoders
                    ) 
                for c in X.columns]
        return [item for sublist in lst if sublist is not None for item in sublist]

    @staticmethod
    def _is_dict(col, col_name):
        """Check if numpy array contains dictionaries, if string attempt to conver to dict"""
        col = col.values
        i=0
        while col[i] is None:  
            i+=1
        col = col[i]
        raise_exception = False
        if isinstance(col, str):
            try:
                ast.literal_eval(col)
                raise_exception = True
            except:
                return False
        if raise_exception:
            raise ValueError("Convert this column to dict before fitting: {0}".format(col_name))
        return isinstance(col, dict)

    @staticmethod
    def _is_list(col, col_name):
        """Check if numpy array contains lists of strings, if string attempt to conver to list"""
        col = col.values
        i=0 
        while col[i] is None:
            i+=1
        col = col[i]
        raise_exception = False
        if isinstance(col, str):
            try:
                ast.literal_eval(col)
                raise_exception = True
            except:
                return False
        if raise_exception:
            raise ValueError("Convert this column to list before fitting: {0}".format(col_name))
        return isinstance(col, list)

    @staticmethod
    def _is_tuple(col, col_name):
        """Check if numpy array contains tuples of strings, if string attempt to conver to tuple"""
        col = col.values
        i=0 
        while col[i] is None:
            i+=1
        col = col[i]
        raise_exception = False
        if isinstance(col, str):
            try:
                ast.literal_eval(col)
                raise_exception = True
            except:
                return False
        if raise_exception:
            raise ValueError("Convert this column to tuple before fitting: {0}".format(col_name))
        return isinstance(col, tuple)

    def _infer_column(self, col_name, X, _default_encoders, thresh=0.10):
        """ Infer encoder type of individual DataFrame column """
        if np.all(X.values == None):
            assert Warning('Column is entirely null: {0}'.format(col_name)) 
            return None       

        is_dict = self._is_dict(X, col_name)
        if is_dict:
            return _default_encoders[self.size]["dict"](col_name)

        is_list = self._is_list(X, col_name)
        if is_list:
            return _default_encoders[self.size]["multihotencoder"](col_name)

        is_tuple = self._is_tuple(X, col_name)
        if is_tuple:
            return _default_encoders[self.size]["multihotencoder"](col_name)
        
        try:
            np.mean(X)
            is_numeric = True
        except:
            is_numeric = False
    
        num_obs = float(len(X))
        pct_unique = len(X.unique()) / num_obs
        is_categorical = pct_unique < thresh
        
        if not is_numeric and not is_categorical:
            return _default_encoders[self.size]["string_vectorizer"](col_name)
        elif is_numeric and not is_categorical:
            return _default_encoders[self.size]["numeric"](col_name)
        else:
            return _default_encoders[self.size]["onehotencoder"](col_name)

    def _feature_indices(self, X):
        """ Save transformed feature vector sizes per transformer """
        projections = [
            transformer.transform(X.head(1))
            for _, transformer, _ 
            in self._iter()]
        
        self.transformer_lengths = [
            len(x[0]) if isinstance(x, list) else x.shape[1]
            for x in projections]

class EncoderizerExtractor(TransformerMixin, BaseEstimator):
    """
    Transformer pass through used for hyperparameter optimization
    in a pipeline
    
    Args:
        encoderizer (fitted Encoderizer instance):
            Encoder from which to slice
        step_names (array-like):
            List of step_names to extract from encoderizer
    """
    def __init__(self, encoderizer, step_names):
        self.encoderizer = encoderizer.extract(step_names)
        
    def fit(self, X, y=None):
        """ Trivial fit method """
        return self

    def transform(self, X):
        """ Extract appropriate transformation steps from Encoderizer """
        return self.encoderizer.transform(X)
