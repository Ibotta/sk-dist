"""
========================================================
Hand-written digits example with distributed grid search
========================================================

This example is roughly the scikit-learn hand-written digits example found here:
https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html

Here we use grid search on the support vector classifier, trying out
various kernels and coefficients.

This totals 750 fits. Our wall time from the sample run 
is about 1.4 seconds. From our spark logs (not shown here) 
running on a cluster with 640 total cores, we see a total 
task time of 7.3 minutes. 

Sample outut run:

Train time: 1.4478580951690674
Best score: 0.981450024203508
-- CV Results --
   param_C param_kernel  mean_test_score
60      10          rbf         0.981450
66      10          rbf         0.980891
51       1          rbf         0.978653
55       1         poly         0.978472
61      10         poly         0.978472
34     0.1         poly         0.978472
28    0.01         poly         0.978472
49       1         poly         0.978472
25    0.01         poly         0.978472
52       1         poly         0.978472
"""
print(__doc__)

import time
import pandas as pd

from sklearn import datasets, svm
from skdist.distribute.search import DistGridSearchCV
from pyspark.sql import SparkSession

# instantiate spark session
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

# the digits dataset
digits = datasets.load_digits()
X = digits["data"]
y = digits["target"]

# create a classifier: a support vector classifier
classifier = svm.SVC(gamma="scale")
param_grid = {
    "C": [0.01, 0.01, 0.1, 1.0, 10.0],
    "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    "kernel": ["rbf", "poly", "sigmoid"],
}
scoring = "f1_weighted"
cv = 10

# hyperparameter optimization
# total fits: 750
start = time.time()
model = DistGridSearchCV(classifier, param_grid, sc=sc, cv=cv, scoring=scoring)
model.fit(X, y)
print("Train time: {0}".format(time.time() - start))
print("Best score: {0}".format(model.best_score_))
results = pd.DataFrame(model.cv_results_).sort_values(
    "mean_test_score", ascending=False
)
print("-- CV Results --")
print(results[["param_C", "param_kernel", "mean_test_score"]].head(10))
