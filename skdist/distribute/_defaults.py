"""
Default feature encoding functions and pipelines
for automated feature transformation.
"""

from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from ..preprocessing import (
    FillNA, VarSelect, CastToType, 
    LabelEncoderPipe, HashingVectorizerChunked
    )
from ..word_embedding import D2VTransformer

def tokenizer(x):
    return x

dict_encoder = lambda c: [(
                "{0}_dict_encoder".format(c),
                Pipeline(steps=[
                    ("var", VarSelect(c)),
                    ("fillna", FillNA({})),
                    ("vec", DictVectorizer())
                ])  
        )]

onehot_encoder = lambda c: [(
                "{0}_onehot".format(c), 
                 Pipeline(steps=[
                     ("var", VarSelect(c)),
                     ("cast", CastToType(type=str)),
                     ("fillna", FillNA("")),
                     ("vec", CountVectorizer(
                         tokenizer=tokenizer, 
                         binary=True,
                         decode_error="ignore"))])
                )]

numeric_encoder = lambda c: [(
                "{0}_scaler".format(c),
                Pipeline(steps=[
                    ("var", VarSelect(c, flatten=False)),
                    ("imputer", SimpleImputer(strategy="median")), 
                    ("scaler", StandardScaler(copy=False))])
                )]

_default_encoders = {
    "small": {
        "string_vectorizer": lambda c: [(
                "{0}_word_vec".format(c), 
                 Pipeline(steps=[
                     ("var", VarSelect(c)),
                     ("fillna", FillNA(" ")), 
                     ("vec", HashingVectorizerChunked(
                         ngram_range=(1,2), 
                         analyzer="word", 
                         decode_error="ignore")),
                     ("var_thresh", VarianceThreshold())])
                )],
        "onehotencoder": onehot_encoder,
        "numeric": numeric_encoder,
        "dict": dict_encoder
    },
    "medium": {
        "string_vectorizer": lambda c: [(
                    "{0}_word_vec".format(c), 
                     Pipeline(steps=[
                         ("var", VarSelect(c)),
                         ("fillna", FillNA(" ")), 
                         ("vec", HashingVectorizerChunked(
                             ngram_range=(1,3), 
                             analyzer="word", 
                             decode_error="ignore")),
                         ("var_thresh", VarianceThreshold())])
                    ),
                    (
                    "{0}_char_vec".format(c), 
                     Pipeline(steps=[
                         ("var", VarSelect(c)),
                         ("fillna", FillNA(" ")), 
                         ("vec", HashingVectorizerChunked(
                             ngram_range=(3,4), 
                             analyzer="char_wb", 
                             decode_error="ignore")),
                         ("var_thresh", VarianceThreshold())])
                    )
                ],
        "onehotencoder": onehot_encoder,
        "numeric": numeric_encoder,
        "dict": dict_encoder
    },
    "large": {
        "string_vectorizer": lambda c: [(
                    "{0}_word_vec".format(c), 
                     Pipeline(steps=[
                         ("var", VarSelect(c)),
                         ("fillna", FillNA(" ")), 
                         ("vec", HashingVectorizerChunked(
                             ngram_range=(1,3), 
                             analyzer="word", 
                             decode_error="ignore")),
                         ("var_thresh", VarianceThreshold())])
                    ),
                    (
                    "{0}_char_vec".format(c), 
                     Pipeline(steps=[
                         ("var", VarSelect(c)),
                         ("fillna", FillNA(" ")), 
                         ("vec", HashingVectorizerChunked(
                             ngram_range=(3,5), 
                             analyzer="char_wb", 
                             decode_error="ignore")),
                         ("var_thresh", VarianceThreshold())])
                    ),
                    (
                    "{0}_word2vec1".format(c), 
                     Pipeline(steps=[
                         ("var", VarSelect(c)),
                         ("fillna", FillNA(" ")), 
                         ("vec", D2VTransformer(
                             size=100, min_count=20, round_n=2)),
                         ("imputer", SimpleImputer(strategy="median"))])
                    ),
                    (
                    "{0}_word2vec2".format(c), 
                     Pipeline(steps=[
                         ("var", VarSelect(c)),
                         ("fillna", FillNA(" ")), 
                         ("vec", D2VTransformer(
                             size=250, min_count=20, round_n=2)),
                         ("imputer", SimpleImputer(strategy="median"))])
                    )
                ],
        "onehotencoder": onehot_encoder,
        "numeric": numeric_encoder,
        "dict": dict_encoder
    }
}
