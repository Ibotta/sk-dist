"""
===================================================================================
Train distributed CV search with a logistic regression on the breast cancer dataset
===================================================================================

In this example we optimize hyperparameters (C) for a logistic regression on the
breast cancer dataset. This is a binary target. We use both grid search and 
randomized search. 

Here the core difference between skdist and sklearn is to use the sparkContext
variable as an argument to the grid search and randomized search class 
instantiation. Under the hood, skdist will then broadcast the training data out
to the executors for each param set, fit the estimator for each param set, return
the cross validation score to the driver for each fit, and finally refit the model 
with the best param set back on the driver.

The final estimators are then nearly identical to a fitted sklearn GridSearchCV
or RandomizedSearchCV estimator as shown by looking at some of their methods
and attributes. 

Finally, all spark objects are removed from the fitted skdist estimator objects
so that these objects are pickle-able as shown.

Here is a sample output run:

-- Grid Search --
Best Score: 0.9925297825837328
Best C: 1.0
  param_C  mean_test_score
0   0.001         0.973818
1    0.01         0.982880
2     0.1         0.989827
3       1         0.992530
4      10         0.992010
5     100         0.990754
DistGridSearchCV(estimator=LogisticRegression(C=1.0, class_weight=None,
                                              dual=False, fit_intercept=True,
                                              intercept_scaling=1,
                                              l1_ratio=None, max_iter=100,
                                              multi_class='warn', n_jobs=None,
                                              penalty='l2', random_state=None,
                                              solver='liblinear', tol=0.0001,
                                              verbose=0, warm_start=False),
                 param_grid={'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
                 partitions='auto', preds=False, sc=None)
-- Randomized Search --
Best Score: 0.9925297825837328
Best C: 1.0
  param_C  mean_test_score
3    0.01         0.982880
2     0.1         0.989827
4       1         0.992530
1      10         0.992010
0     100         0.990754
DistRandomizedSearchCV(estimator=LogisticRegression(C=1.0, class_weight=None,
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
                       param_distributions={'C': [0.001, 0.01, 0.1, 1.0, 10.0,
                                                  100.0]},
                       partitions='auto', preds=False, sc=None)
"""
print(__doc__)

import pickle
import pandas as pd
import numpy as np

from skdist.distribute.search import DistGridSearchCV, DistRandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from pyspark.sql import SparkSession

# spark session initialization
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

# sklearn variables
Cs = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
cv = 5
n_iter = 5
scoring = "roc_auc"
solver = "liblinear"

# load sample data (binary target)
data = load_breast_cancer()
X = data["data"]
y = data["target"]

### distributed grid search
model = DistGridSearchCV(
    LogisticRegression(solver=solver), dict(C=Cs), sc, cv=cv, scoring=scoring
)
# distributed fitting with spark
model.fit(X, y)
# predictions on the driver
preds = model.predict(X)
probs = model.predict_proba(X)

# results
print("-- Grid Search --")
print("Best Score: {0}".format(model.best_score_))
print("Best C: {0}".format(model.best_estimator_.C))
result_data = pd.DataFrame(model.cv_results_)[["param_C", "mean_test_score"]]
print(result_data.sort_values("param_C"))
print(pickle.loads(pickle.dumps(model)))

### distributed randomized search
param_dist = dict(C=[])
model = DistRandomizedSearchCV(
    LogisticRegression(solver=solver),
    dict(C=Cs),
    sc,
    cv=cv,
    scoring=scoring,
    n_iter=n_iter,
)
# distributed fitting with spark
model.fit(X, y)
# predictions on the driver
preds = model.predict(X)
probs = model.predict_proba(X)

# results
print("-- Randomized Search --")
print("Best Score: {0}".format(model.best_score_))
print("Best C: {0}".format(model.best_estimator_.C))
result_data = pd.DataFrame(model.cv_results_)[["param_C", "mean_test_score"]]
print(result_data.sort_values("param_C"))
print(pickle.loads(pickle.dumps(model)))
