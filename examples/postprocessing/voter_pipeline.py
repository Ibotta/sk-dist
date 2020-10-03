"""
=========================================================================================
Predict with a simple voting classifier using models trained with distributed grid search
=========================================================================================

In this example we train two models with distributed grid search using 
two categories from the 20 newsgroups dataset:
- LogisticRegression
- RandomForestClassifier

The text data features are encoded with the Encoderizer. The fitted
encoderizer and the assembled SimpleVoter are then combined in a 
Pipeline.

Predictions can then be made with 'hard' or 'soft' voting with the 
entire pipeline.
"""
print(__doc__)

import pandas as pd

from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from pyspark.sql import SparkSession
from skdist.postprocessing import SimpleVoter
from skdist.distribute.search import DistGridSearchCV
from skdist.distribute.encoder import Encoderizer

# instantiate spark session
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

# load two categories of 20 newsgroups dataset
categories = ["alt.atheism", "talk.religion.misc"]
dataset = fetch_20newsgroups(
    shuffle=True,
    random_state=1,
    remove=("headers", "footers", "quotes"),
    categories=categories,
)

# variables
cv = 5
scoring = "roc_auc"
solver = "liblinear"
limit = 1000

# convert training data to pandas
df = pd.DataFrame({"text": dataset["data"]})
df = df[:limit]
dataset["target"] = dataset["target"][:limit]

# fit a small encoder
encoder = Encoderizer(size="small")
X_t = encoder.fit_transform(df)

# train logistic regression
lr = DistGridSearchCV(
    LogisticRegression(solver="liblinear"),
    dict(C=[0.1, 1.0, 10.0]),
    sc,
    scoring=scoring,
    cv=cv,
)
lr.fit(X_t, dataset["target"])

# train random forest
rf = DistGridSearchCV(
    RandomForestClassifier(n_estimators=10),
    dict(max_depth=[5, 10]),
    sc,
    scoring=scoring,
    cv=cv,
)
rf.fit(X_t, dataset["target"])

# assemble voter and pipeline
voter = SimpleVoter([("lr", lr), ("rf", rf)], classes=model.classes_, voting="hard")
model = Pipeline(steps=[("vec", encoder), ("clf", voter)])

# make predictions
preds = model.predict(df)
