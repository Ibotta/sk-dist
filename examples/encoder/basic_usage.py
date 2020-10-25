"""
==========================================================================
Fit a text encoder using Encoderizer and train classifiers on 20newsgroups 
==========================================================================

In this example we use Encoderizer to transform training text
features into a sparse matrix for training a simple classifier. 

The Encoderizer will vectorizer the text using various methods according
to the size input. A size of small will stick to simple hashing vectorized
word tokens. A size of medium will include character tokens as well. A
size of large will include hashing vectorized word tokens and character tokens
with more n-grams.

At transform time, the Encoderizer functions much like a sklearn
FeatureUnion pipeline.

In this case, we'll look at a small, medium and large sized Encoderizer. 
Scores improve from small to medium and then do worse for large.

Here is a sample output run:

0.3795335716355121
0.46713058755408793
0.45031619208147355
"""
print(__doc__)

import pandas as pd

from skdist.distribute.search import DistGridSearchCV
from skdist.distribute.encoder import Encoderizer
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_20newsgroups
from pyspark.sql import SparkSession

# spark session initialization
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

# load 20newsgroups dataset
dataset = fetch_20newsgroups(
    shuffle=True, random_state=1, remove=("headers", "footers", "quotes")
)

# variables
Cs = [0.1, 1.0, 10.0]
cv = 5
scoring = "f1_weighted"
solver = "liblinear"

# convert training data to pandas
df = pd.DataFrame({"text": dataset["data"]})
df = df[:1000]
dataset["target"] = dataset["target"][:1000]

# fit a small encoder and train classifier
encoder = Encoderizer(size="small")
X_t = encoder.fit_transform(df)
model = DistGridSearchCV(
    LogisticRegression(solver=solver, multi_class="auto"),
    dict(C=Cs),
    sc,
    scoring=scoring,
    cv=cv,
)
model.fit(X_t, dataset["target"])
print(model.best_score_)

# fit a medium encoder and train classifier
encoder = Encoderizer(size="medium")
X_t = encoder.fit_transform(df)
model = DistGridSearchCV(
    LogisticRegression(solver=solver, multi_class="auto"),
    dict(C=Cs),
    sc,
    scoring=scoring,
    cv=cv,
)
model.fit(X_t, dataset["target"])
print(model.best_score_)

# fit a large encoder and train classifier
encoder = Encoderizer(size="large")
X_t = encoder.fit_transform(df)
model = DistGridSearchCV(
    LogisticRegression(solver=solver, multi_class="auto"),
    dict(C=Cs),
    sc,
    scoring=scoring,
    cv=cv,
)
model.fit(X_t, dataset["target"])
print(model.best_score_)
