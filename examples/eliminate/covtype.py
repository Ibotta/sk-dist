"""
===============================================================================
Fit a random forest on the covtype dataset with distributed feature elimination 
===============================================================================

The covtype dataset has 54 dense features:

    =================   ============
    Classes                        7
    Samples total             581012
    Dimensionality                54
    Features                     int
    =================   ============

This example shows how to elimiate features that 
don't improve a cross validation score using `f1_weighted`. 
Here we see only 21 featues, less than half being selected,
resulting in a better scoring model than using all of the
given features. This results in non-trival lift over
the trival feature set (all of them).

This runs in 4.58 minutes. It would take 5+ hours if run
recursively with no parallelization.

Here is an example output run:

-- skdist DistFeatureEliminator --
Fit Time: 275.2176
N Features Selected: 21
Best Score: 0.640838396611427
Score w/ All Features: 0.6257725436874217
Feature Elimination Lift: 0.015065852924005307
"""
print(__doc__)

import time
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from skdist.distribute.eliminate import DistFeatureEliminator
from sklearn.datasets import fetch_covtype
from pyspark.sql import SparkSession

# instantiate spark session
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

# params
scoring = "f1_weighted"
cv = 5
min_features_to_select = 10

# load data and define base classifier
X, y = fetch_covtype(return_X_y=True)
clf = RandomForestClassifier(n_estimators=100, max_depth=10)

# eliminate features, keeping at least 10
start = time.time()
model = DistFeatureEliminator(
    clf, sc=sc, scoring=scoring, cv=cv, min_features_to_select=min_features_to_select
)
model.fit(X, y)
print("-- skdist DistFeatureEliminator --")
print("Fit Time: {0}".format(round(time.time() - start, 4)))
print("N Features Selected: {0}".format(model.n_features_))
print("Best Score: {0}".format(model.best_score_))
print("Score w/ All Features: {0}".format(model.scores_[0]))
print("Feature Elimination Lift: {0}".format(model.best_score_ - model.scores_[0]))
