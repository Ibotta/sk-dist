"""
Gensim model transformers with custom vector
aggregation techniques. Implemented as scikit-learn
transformers compatible with scikit-learn pipelines.
"""

import re
import numpy as np

from sklearn.exceptions import NotFittedError
from sklearn.base import TransformerMixin, BaseEstimator

class GensimRequired(ImportError):
    pass

_GENSIM_INSTALLED = None

def _is_gensim_installed():
    global _GENSIM_INSTALLED
    if _GENSIM_INSTALLED is None:
        try:
            from gensim.models import Word2Vec
            _GENSIM_INSTALLED = True
        except ImportError:
            _GENSIM_INSTALLED = False

    if _GENSIM_INSTALLED:
        from gensim.models import Word2Vec
        return True
    else:
        return False

def _tokenize_simple(text):
    """ Simple text tokenizer for default tokenization """
    return re.sub("[^a-zA-z]", " ", text.lower()).split()

class BaseGensimTransformer(TransformerMixin, BaseEstimator):
    """
    Base sklearn wrapper for gensim models. See gensim.models for parameter details.

    Args:
        gensim_model_class (object): Gensim model class
        agg_func (function): Function used to aggregate word vectors
        tokenizer (function): Text tokenization function.
        round_n (int): Number of decimals to round transformed vector
    """
    def __init__(self, gensim_model_class, agg_func=None, tokenizer=None, round_n=None):
        self.gensim_model_class = gensim_model_class
        self.agg_func = agg_func
        self.tokenizer = tokenizer
        self.round_n = round_n
        
    def fit(self, X, y=None):
        """
        Fit the model gensim model.
        Args: 
            X (array-like): Iterable of string documents.
        """
        sentences = self._tokenize(X)
        self.gensim_model = self.gensim_model_class(
            sentences, **self.model_kwargs
            )
        self.gensim_model = self.gensim_model.wv
        return self
        
    def transform(self, X):
        """ 
        Transform input data with gensim model
        Args:
            X (array-like): Iterable of string documents.
        """
        if self.gensim_model is None:
            raise NotFittedError(
                "This model has not been fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )

        if self.agg_func is None:
            self.agg_func = np.nanmean
        vectors = [
            self._agged_to_list(
                self.agg_func([self._get_vec(y) for y in x], 
                axis=0)) 
            for x in self._tokenize(X)]
        return vectors
        
    def _get_vec(self, word):
        """ Get word vector given word string """
        try:
            arr = self.gensim_model[word]
            if self.round_n is not None:
                return np.round(arr, self.round_n)
            else:
                return arr
        except:
            return np.array([np.nan]*self.size)
            
    def _tokenize(self, X):
        """ Tokenize iterable of text strings """
        if self.tokenizer is None:
            return [_tokenize_simple(x) for x in X]
        else:
            return [self.tokenizer(x) for x in X]
            
    def _agged_to_list(self, x):
        """ Replace nans with array of nans """
        try:
            return list(x)
        except:
            return np.array([np.nan]*self.size)
            
class D2VTransformer(BaseGensimTransformer):
    """
    Doc2Vec sklearn transformer using underlying gensim
    Word2Vec model. Fits Word2Vec model
    on a corpus of documents. Transforms documents by
    tokenizing and applying `agg_func` to the word vectors for each
    token. Out-of-vocab tokens will return NaN values
    so documents with no tokens in the fitted vocabulary will
    result in an array of NaN values at transformation time.

    Consider coupling with a sklearn `Imputer` transformer to 
    handle NaNs when using in a prediction pipeline.

    Args:
        gensim_model_class (object): Gensim model class
        agg_func (function): Function used to aggregate word vectors
        tokenizer (function): Text tokenization function.
        round_n (int): Number of decimals to round transformed vector
        **kwargs : Word2Vec model keyword arguments.
    """
    def __init__(self, agg_func=None, tokenizer=None, round_n=None, size=100, alpha=0.025, window=5, 
            min_count=5, max_vocab_size=None, sample=1e-3, seed=1, workers=3, 
            min_alpha=0.0001, sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
            trim_rule=None, sorted_vocab=1, batch_words=10000):

        if not _is_gensim_installed():
            raise ModuleNotFoundError("Module gensim not found")
        from gensim.models import Word2Vec

        self.gensim_model_class = Word2Vec
        self.agg_func = agg_func
        self.tokenizer = tokenizer
        self.round_n = round_n
        self.size = size
        self.model_kwargs = dict(
            size=size,
            alpha=alpha,
            window=window,
            min_count=min_count,
            max_vocab_size=max_vocab_size,
            sample=sample,
            seed=seed,
            workers=workers,
            min_alpha=min_alpha,
            sg=sg,
            hs=hs,
            negative=negative,
            cbow_mean=int(cbow_mean),
            hashfxn=hashfxn,
            iter=iter,
            null_word=null_word,
            trim_rule=trim_rule,
            sorted_vocab=sorted_vocab,
            batch_words=batch_words)
            
