"""
Default feature encoding functions and pipelines
for automated feature transformation.
"""

from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer

from ..preprocessing import (
    ImputeNull, SelectField, FeatureCast, 
    LabelEncoderPipe, HashingVectorizerChunked,
    MultihotEncoder
    )

def tokenizer(x):
    """ Trivial tokenizer """
    return x

dict_encoder = lambda c: [(
                "{0}_dict_encoder".format(c),
                Pipeline(steps=[
                    ("var", SelectField(cols=[c], single_dimension=True)),
                    ("fillna", ImputeNull({})),
                    ("vec", DictVectorizer())])  
                )]

onehot_encoder = lambda c: [(
                "{0}_onehot".format(c), 
                 Pipeline(steps=[
                     ("var", SelectField(cols=[c], single_dimension=True)),
                     ("cast", FeatureCast(cast_type=str)),
                     ("fillna", ImputeNull("")),
                     ("vec", CountVectorizer(
                         token_pattern=None,
                         tokenizer=tokenizer, 
                         binary=True,
                         decode_error="ignore"))])
                )]

multihot_encoder = lambda c: [(
                "{0}_multihot".format(c), 
                Pipeline(steps=[
                   ("var", SelectField(cols=[c], single_dimension=True)),
                   ("fillna", ImputeNull([])),
                   ("vec", MultihotEncoder())])
                )]

numeric_encoder = lambda c: [(
                "{0}_scaler".format(c),
                Pipeline(steps=[
                    ("var", SelectField(cols=[c])),
                    ("imputer", SimpleImputer(strategy="median")), 
                    ("scaler", StandardScaler(copy=False))])
                )]

_default_encoders = {
    "small": {
        "string_vectorizer": lambda c: [(
                "{0}_word_vec".format(c), 
                 Pipeline(steps=[
                     ("var", SelectField(cols=[c], single_dimension=True)),
                     ("fillna", ImputeNull(" ")), 
                     ("vec", HashingVectorizerChunked(
                         ngram_range=(1,2), 
                         analyzer="word", 
                         decode_error="ignore")),
                     ("var_thresh", VarianceThreshold())])
                )],
        "onehotencoder": onehot_encoder,
        "multihotencoder": multihot_encoder,
        "numeric": numeric_encoder,
        "dict": dict_encoder
    },
    "medium": {
        "string_vectorizer": lambda c: [(
                    "{0}_word_vec".format(c), 
                     Pipeline(steps=[
                         ("var", SelectField(cols=[c], single_dimension=True)),
                         ("fillna", ImputeNull(" ")), 
                         ("vec", HashingVectorizerChunked(
                             ngram_range=(1,3), 
                             analyzer="word", 
                             decode_error="ignore")),
                         ("var_thresh", VarianceThreshold())])
                    ),
                    (
                    "{0}_char_vec".format(c), 
                     Pipeline(steps=[
                         ("var", SelectField(cols=[c], single_dimension=True)),
                         ("fillna", ImputeNull(" ")), 
                         ("vec", HashingVectorizerChunked(
                             ngram_range=(3,4), 
                             analyzer="char_wb", 
                             decode_error="ignore")),
                         ("var_thresh", VarianceThreshold())])
                    )
                ],
        "onehotencoder": onehot_encoder,
        "multihotencoder": multihot_encoder,
        "numeric": numeric_encoder,
        "dict": dict_encoder
    },
    "large": {
        "string_vectorizer": lambda c: [(
                    "{0}_word_vec".format(c), 
                     Pipeline(steps=[
                         ("var", SelectField(cols=[c], single_dimension=True)),
                         ("fillna", ImputeNull(" ")), 
                         ("vec", HashingVectorizerChunked(
                             ngram_range=(1,3), 
                             analyzer="word", 
                             decode_error="ignore")),
                         ("var_thresh", VarianceThreshold())])
                    ),
                    (
                    "{0}_char_vec".format(c), 
                     Pipeline(steps=[
                         ("var", SelectField(cols=[c], single_dimension=True)),
                         ("fillna", ImputeNull(" ")), 
                         ("vec", HashingVectorizerChunked(
                             ngram_range=(2,5), 
                             analyzer="char_wb", 
                             decode_error="ignore")),
                         ("var_thresh", VarianceThreshold())])
                    )
                ],
        "onehotencoder": onehot_encoder,
        "multihotencoder": multihot_encoder,
        "numeric": numeric_encoder,
        "dict": dict_encoder
    }
}
