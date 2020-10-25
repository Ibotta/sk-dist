"""
================================================================
Approximate RFECV using sk-dist with 46x performance improvement
================================================================

In this example, we use `make_classification` to build a synthetic
training data set. It has binary labels and 100k observations.
The training data has 40 total features with 12 predictive features.
We use two feature elimination algorithms to try to find the 12
predictive features, ultimately training the best model
according to cross validation scores.

The two algorithms ultimately find the 12 predictive features and
train the best model. The difference is in the runtime.

DistFeatureEliminator - small spark cluster: runtime 3 mins
RFECV - one core on one machine: runtime 2.4 hours

Example output run:

-- Distributed Feature Elimination --
Elapsed Time: 182.6893
Score w/ All Features: 0.8391530109906709
Number of Features Selected: 12
Best Score: 0.8893020795290841
-- RFE CV --
Elapsed Time: 8538.8849
Score w/ All Features: 0.8392805035454698
Number of Features Selected: 12
Best Score: 0.889048190663663
"""
print(__doc__)

import time
import numpy as np

from skdist.distribute.eliminate import DistFeatureEliminator
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from pyspark.sql import SparkSession

# spark session initialization
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

# parameters
n_estimators = 100
max_depth = 20
scoring = "roc_auc"
min_features_to_select = 6
cv = 5

# build synthetic training data
X, y = make_classification(
    n_samples=100000,
    n_features=40,
    n_informative=12,
    n_classes=2,
    n_clusters_per_class=20,
    random_state=5,
)

# train distributed feature eliminator
start = time.time()
fe = DistFeatureEliminator(
    RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth),
    scoring=scoring,
    min_features_to_select=min_features_to_select,
    cv=cv,
    sc=sc,
)
fe.fit(X, y)
print("-- Distributed Feature Elimination --")
print("Elapsed Time: {0}".format(round((time.time() - start), 4)))
print("Score w/ All Features: {0}".format(fe.scores_[0]))
print("Number of Features Selected: {0}".format(fe.n_features_))
print("Best Score: {0}".format(fe.best_score_))

# train recursive cross validated feature eliminator
start = time.time()
rfe = RFECV(
    RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth),
    scoring=scoring,
    min_features_to_select=min_features_to_select,
    cv=cv,
)
rfe.fit(X, y)
print("-- RFE CV --")
print("Elapsed Time: {0}".format(round((time.time() - start), 4)))
print("Score w/ All Features: {0}".format(rfe.grid_scores_[-1]))
print("Number of Features Selected: {0}".format(rfe.n_features_))
print("Best Score: {0}".format(np.max(rfe.grid_scores_)))
