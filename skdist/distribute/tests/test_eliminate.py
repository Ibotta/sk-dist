"""
Test feature eliminator classes
"""

import numpy as np

from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

try:
    from skdist.distribute import eliminate
    from skdist.distribute.eliminate import DistFeatureEliminator 
    _import_error = None
except Exception as e:
    _import_error = e

def test_import():
    assert _import_error == None

def test_fit():
    X,y = datasets.load_iris(return_X_y=True)
    fe = DistFeatureEliminator(
        LogisticRegression(solver="liblinear", multi_class="auto"), 
        scoring="f1_weighted", min_features_to_select=3, cv=3
        )
    fe.fit(X,y)
    assert np.allclose(fe.best_features_, [1,2,3])

def test_score():
    X,y = datasets.load_iris(return_X_y=True)
    fe = DistFeatureEliminator(
        LogisticRegression(solver="liblinear", multi_class="auto"),
        scoring="f1_weighted", min_features_to_select=2, cv=3
        )
    fe.fit(X,y)
    assert round(fe.score(X,y),2) == 0.95

def test_predict():
    X,y = datasets.load_iris(return_X_y=True)
    fe = DistFeatureEliminator(
        LogisticRegression(solver="liblinear", multi_class="auto"),
        scoring="f1_weighted", min_features_to_select=1, cv=3, step=2
        )
    fe.fit(X,y)
    assert fe.predict(X)[0] == 0

def test_sparse():
    X,y = datasets.load_iris(return_X_y=True)
    fe = DistFeatureEliminator(
        LogisticRegression(solver="liblinear", multi_class="auto"),
        scoring="f1_weighted", min_features_to_select=3, cv=3
        )
    fe.fit(csr_matrix(X),y)
    assert fe.predict(X)[0] == 0
