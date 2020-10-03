"""
=============================================================
Train distributed multiclass strategies on the digits dataset
=============================================================

In this example we fit two different multiclass strategies on the digits
dataset with distributed training using spark. The two strategies
are the popular one-vs-rest strategy, as well as the one-vs-one
strategy.

Here the core difference between skdist and sklearn is to use the sparkContext
variable as an argument to the one-vs-rest and one-vs-one class
instantiation. Under the hood, skdist will then broadcast the training data and
specific binary target data out to the executors for each binary model, 
fit the estimator for each binary model, collect each binary model on the
driver, and appropriately store those binary models within the fitted
estimator object to conform to the predict methods of the sklearn multiclass
meta-estimators.

The final estimators are then nearly identical to a fitted sklearn OneVsRestClassifier
or OneVsOneClassifier estimator as shown by looking at some of their methods
and attributes.

Finally, all spark objects are removed from the fitted skdist estimator objects
so that these objects are pickle-able as shown.

In this example, one-vs-one outperforms. It is an interesting but underutilized
multiclass strategy, largely due to the number of fits required at train time
time (n_classes * (n_classes - 1) / 2). In this case at n_classes = 10, it only requires 
45 fits. However, for many real world problems as n_samples and n_classes get
large, this becomes less practical. Distributing this training with spark can
drastically speed up that train time, making this a more feasible multiclass 
strategy.
 
Here is a sample output run:

-- One Vs Rest --
Weighted F1: 0.9588631625829608
Precision: 0.9609833792125458
Recall: 0.9583333333333334
DistOneVsRestClassifier(estimator=LogisticRegression(C=1.0, class_weight=None,
                                                     dual=False,
                                                     fit_intercept=True,
                                                     intercept_scaling=1,
                                                     l1_ratio=None,
                                                     max_iter=100,
                                                     multi_class='warn',
                                                     n_jobs=None, penalty='l2',
                                                     random_state=None,
                                                     solver='liblinear',
                                                     tol=0.0001, verbose=0,
                                                     warm_start=False),
                        max_negatives=None, method='ratio', mlb_override=False,
                        n_splits=1, norm=None, partitions='auto',
                        random_state=None, sc=None)
-- One Vs One --
Weighted F1: 0.9805258819058627
Precision: 0.980952380952381
Recall: 0.9805555555555555
DistOneVsOneClassifier(estimator=LogisticRegression(C=1.0, class_weight=None,
                                                    dual=False,
                                                    fit_intercept=True,
                                                    intercept_scaling=1,
                                                    l1_ratio=None, max_iter=100,
                                                    multi_class='warn',
                                                    n_jobs=None, penalty='l2',
                                                    random_state=None,
                                                    solver='liblinear',
                                                    tol=0.0001, verbose=0,
                                                    warm_start=False),
                       partitions='auto', sc=None)
"""
print(__doc__)

import pickle
import pandas as pd
import numpy as np

from skdist.distribute.multiclass import DistOneVsRestClassifier, DistOneVsOneClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from pyspark.sql import SparkSession

# spark session initialization
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

# variables
scoring_average = "weighted"
solver = "liblinear"
test_size = 0.2

# load sample data (binary target)
data = load_digits()
X = data["data"]
y = data["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=10
)

### distributed one vs rest
model = DistOneVsRestClassifier(LogisticRegression(solver=solver), sc)
# distributed fitting with spark
model.fit(X_train, y_train)
# predictions on the driver
preds = model.predict(X_test)
probs = model.predict_proba(X_test)

# results
print("-- One Vs Rest --")
print("Weighted F1: {0}".format(f1_score(y_test, preds, average=scoring_average)))
print("Precision: {0}".format(precision_score(y_test, preds, average=scoring_average)))
print("Recall: {0}".format(recall_score(y_test, preds, average=scoring_average)))
print(pickle.loads(pickle.dumps(model)))

### distributed one vs one
model = DistOneVsOneClassifier(LogisticRegression(solver=solver), sc)
# distributed fitting with spark
model.fit(X_train, y_train)
# predictions on the driver
preds = model.predict(X_test)

# results
print("-- One Vs One --")
print("Weighted F1: {0}".format(f1_score(y_test, preds, average=scoring_average)))
print("Precision: {0}".format(precision_score(y_test, preds, average=scoring_average)))
print("Recall: {0}".format(recall_score(y_test, preds, average=scoring_average)))
print(pickle.loads(pickle.dumps(model)))
