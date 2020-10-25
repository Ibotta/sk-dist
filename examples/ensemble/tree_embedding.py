"""
===============================================================
Encode circles data with tree embedding and compare classifiers
===============================================================

Here we look at an example modelled of of scikit-learn's hashing
feature transformation example:
https://scikit-learn.org/stable/auto_examples/ensemble/plot_random_forest_embedding.html

We use more training data and more trees in the extra trees
ensemble. We also score the estimators, both naive bayes and
extra frees on both the raw feature data and the tree embedding
transformed data.

This illustrates the power of using tree embedding for hashing
feature transformation. Both the naive bayes and extra trees
classifiers do better with the transformed data. Of particular
interest is the naive bayes model performing no better than 
random on the original data but very high scoring with the
hashing transformed data.

Here is a sample output run:

Naive Bayes -- Transformed: 0.9733504
Naive Bayes -- Original: 0.4964787
Extra Trees -- Transformed: 0.98369
Extra Trees -- Original: 0.9469593
"""
print(__doc__)

import numpy as np

from sklearn.datasets import make_circles
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB
from skdist.distribute.ensemble import DistRandomTreesEmbedding
from pyspark.sql import SparkSession

# instantiate spark session
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

# make a synthetic dataset
X, y = make_circles(n_samples=10000, factor=0.5, random_state=0, noise=0.15)

# use DistRandomTreesEmbedding to transform data
hasher = DistRandomTreesEmbedding(n_estimators=1000, random_state=0, max_depth=3, sc=sc)
X_transformed = hasher.fit_transform(X)

# score a Naive Bayes classifier on the original and transformed data
nb = BernoulliNB()
print(
    "Naive Bayes -- Transformed: {0}".format(
        np.mean(cross_val_score(nb, X_transformed, y, cv=5, scoring="roc_auc"))
    )
)
print(
    "Naive Bayes -- Original: {0}".format(
        np.mean(cross_val_score(nb, X, y, cv=5, scoring="roc_auc"))
    )
)

# score an Extra Trees classifier on the original and transformed data
trees = ExtraTreesClassifier(max_depth=3, n_estimators=10, random_state=0)
print(
    "Extra Trees -- Transformed: {0}".format(
        np.mean(cross_val_score(trees, X_transformed, y, cv=5, scoring="roc_auc"))
    )
)
print(
    "Extra Trees -- Original: {0}".format(
        np.mean(cross_val_score(trees, X, y, cv=5, scoring="roc_auc"))
    )
)
