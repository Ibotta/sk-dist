"""
===================================================================
Nest sk-dist estimators to fit models on `make_classification` data
===================================================================

In this example, we show nesting of sk-dist estimator. An important
note with nesting: sk-dist meta-estimators need to be used to 
nest other sk-dist meta-estimators. Using GridSearchCV from sklearn
with a base estimator of DistRandomForestClassifier will not work.
Handling for this is built into sk-dist, so that appropriate steps
are taken to deal with the sparkContext attribute which does not
like to be cloned with sklearn cloning behavior.

Meta-estimators with sk-dist will mirror their sklearn counterparts
when sc=None as shown in this example. Where you input the sc input 
variable will determine which nesting step uses spark. 

We try three different nesting scenarios:
DistGridSearchCV -> DistOneVsRestClassifier
DistGridSearchCV -> DistOneVsOneClassifier
DistGridSearchCV -> DistRandomForestClassifier

This is an interesting example where we first see the power
of the less popular one-vs-one strategy vs the more popular
one-vs-rest strategy. Then finally a random forest wins out
with native multiclass behavior.

Here is a sample output run:

   mean_test_score param_estimator__C
0         0.345504               0.01
1         0.342376                0.1
2         0.341867                  1
3         0.341769                 10
4         0.341758                 20
5         0.341758                 50
   mean_test_score param_estimator__C
0         0.578317               0.01
1         0.600177                0.1
2         0.604638                  1
3         0.605436                 10
4         0.605495                 20
5         0.605522                 50
   mean_test_score param_max_depth
0         0.597794              10
1         0.677944              20
2         0.675356            None
"""
print(__doc__)

import pandas as pd

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from skdist.distribute.multiclass import DistOneVsRestClassifier, DistOneVsOneClassifier
from skdist.distribute.search import DistGridSearchCV
from skdist.distribute.ensemble import DistRandomForestClassifier
from pyspark.sql import SparkSession

# instantiate spark session
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

# params
scoring = "f1_weighted"
cv = 5
params = [0.01, 0.1, 1.0, 10.0, 20.0, 50.0]

# create dataset
X, y = make_classification(
    n_samples=100000,
    n_features=40,
    n_informative=36,
    n_redundant=1,
    n_repeated=1,
    n_classes=40,
    n_clusters_per_class=1,
    random_state=5,
)

# one nested example
model = DistGridSearchCV(
    DistOneVsRestClassifier(LogisticRegression(solver="liblinear"), sc=sc),
    {"estimator__C": params},
    cv=cv,
    scoring=scoring,
)
model.fit(X, y)
print(pd.DataFrame(model.cv_results_)[["mean_test_score", "param_estimator__C"]])

# another nested example
model = DistGridSearchCV(
    DistOneVsOneClassifier(LogisticRegression(solver="liblinear"), sc=sc),
    {"estimator__C": params},
    cv=cv,
    scoring=scoring,
)
model.fit(X, y)
print(pd.DataFrame(model.cv_results_)[["mean_test_score", "param_estimator__C"]])

# a final nested example
model = DistGridSearchCV(
    DistRandomForestClassifier(sc=sc, n_estimators=100),
    {"max_depth": [10, 20, None], "n_estimators": [100]},
    cv=cv,
    scoring=scoring,
)
model.fit(X, y)
print(pd.DataFrame(model.cv_results_)[["mean_test_score", "param_max_depth"]])
