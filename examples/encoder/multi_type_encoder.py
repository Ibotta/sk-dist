"""
=============================================================================
Fit a multi-type encoder using Encoderizer and train classifiers on fake data
=============================================================================

In this example we use Encoderizer to transform text, categorical, numeric, 
dict, and list features into a sparse matrix for training a simple classifier. 
The highlighted point here is not the fitted model, but the ability to 
specify the types of encoders needed for each feature in the training set.

The Encoderizer will vectorize the features according to the encoding types
specified in the config. This example highlights the complete current set
of encoder options: string_vectorizer, onehotencoder, multihotencoder,
numeric, and dict.

At transform time, the Encoderizer functions much like a sklearn
FeatureUnion pipeline.

Here is a sample output run:

['text_col_word_vec', 'categorical_str_col_onehot', 'categorical_int_col_onehot', 'numeric_col_scaler', 'dict_col_dict_encoder', 'multilabel_col_multihot']
1.0
"""
print(__doc__)

import pandas as pd

from skdist.distribute.encoder import Encoderizer
from skdist.distribute.search import DistGridSearchCV
from sklearn.linear_model import LogisticRegression
from pyspark.sql import SparkSession

# spark session initialization
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

# create some data
text = [
    "this is a text encoding example",
    "more random text for the example",
    "even more random text",
]

df = pd.DataFrame(
    {
        "text_col": text * 4,
        "categorical_str_col": ["control", "treatment", "control"] * 4,
        "categorical_int_col": [0, 1, 2] * 4,
        "numeric_col": [5, 22, 69] * 4,
        "dict_col": [{"a": 4}, {"b": 1}, {"c": 3}] * 4,
        "multilabel_col": [[1, 2], [1, 3], [2]] * 4,
        "target": [1, 0, 1] * 4,
    }
)

# define encoder config
encoder_config = {
    "text_col": "string_vectorizer",
    "categorical_str_col": "onehotencoder",
    "categorical_int_col": "onehotencoder",
    "numeric_col": "numeric",
    "dict_col": "dict",
    "multilabel_col": "multihotencoder",
}

# variables
Cs = [0.1, 1.0, 10.0]
cv = 5
scoring = "f1_weighted"
solver = "liblinear"

# instantiate encoder with encoder_config, fit/transform on data
encoder = Encoderizer(size="small", config=encoder_config)
df_transformed = encoder.fit_transform(df)
print([i[0] for i in encoder.transformer_list])

# define and fit model
model = DistGridSearchCV(
    LogisticRegression(solver=solver, multi_class="auto"),
    dict(C=Cs),
    sc,
    scoring=scoring,
    cv=cv,
)

model.fit(df_transformed, df["target"])
print(model.best_score_)
