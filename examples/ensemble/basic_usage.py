"""
===============================================================
Train distributed decision forests on the breast cancer dataset
===============================================================

In this example we fit two types of decision tree ensembles on
the breast cancer dataset as binary classifiers. These are the
popular random forest classifier and the extra randomized trees
classifier. While the number of decision tree estimators in these
examples is small, skdist allows very large numbers of trees
to be trained in parallel with spark.

Here the core difference between skdist and sklearn is to use the sparkContext
variable as an argument to the random forest and extra trees class
instantiation. Under the hood, skdist will then broadcast the training data out 
to the executors, fit decision trees out on the estimators, 
collect the fitted trees back to the driver, and appropriately store those 
fitted trees within the fitted estimator object to conform to the predict 
methods of the sklearn ensemble meta-estimators.

The final estimators are then nearly identical to a fitted sklearn RandomForestClassifier
or ExtraTreesClassifier estimator as shown by looking at some of their methods
and attributes.

Finally, all spark objects are removed from the fitted skdist estimator objects
so that these objects are pickle-able as shown.

Here is a sample output run:

-- Random Forest --
ROC AUC: 0.9970940170940171
Weighted F1: 0.9864864864864865
Precision: 1.0
Recall: 0.9733333333333334
DistRandomForestClassifier(partitions='auto', sc=None)
-- Extra Trees --
ROC AUC: 0.9955555555555555
Weighted F1: 0.9798657718120806
Precision: 0.9864864864864865
Recall: 0.9733333333333334
DistExtraTreesClassifier(partitions='auto', sc=None)
"""
print(__doc__)

import pickle
import pandas as pd
import numpy as np

from skdist.distribute.ensemble import (
    DistRandomForestClassifier,
    DistExtraTreesClassifier,
)
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from pyspark.sql import SparkSession

# spark session initialization
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

# variables
cv = 5
test_size = 0.2
max_depth = None
n_estimators = 100

# load sample data (binary target)
data = load_breast_cancer()
X = data["data"]
y = data["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=10
)

### distributed random forest
model = DistRandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    sc=sc,
)
# distributed fitting with spark
model.fit(X_train, y_train)
# predictions on the driver
preds = model.predict(X_test)
probs = model.predict_proba(X_test)

# results
print("-- Random Forest --")
print("ROC AUC: {0}".format(roc_auc_score(y_test, probs[:, 1])))
print("Weighted F1: {0}".format(f1_score(y_test, preds)))
print("Precision: {0}".format(precision_score(y_test, preds)))
print("Recall: {0}".format(recall_score(y_test, preds)))
print(pickle.loads(pickle.dumps(model)))

### distributed extra trees
model = DistExtraTreesClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    sc=sc,
)
# distributed fitting with spark
model.fit(X_train, y_train)
# predictions on the driver
preds = model.predict(X_test)
probs = model.predict_proba(X_test)

# results
print("-- Extra Trees --")
print("ROC AUC: {0}".format(roc_auc_score(y_test, probs[:, 1])))
print("Weighted F1: {0}".format(f1_score(y_test, preds)))
print("Precision: {0}".format(precision_score(y_test, preds)))
print("Recall: {0}".format(recall_score(y_test, preds)))
print(pickle.loads(pickle.dumps(model)))
