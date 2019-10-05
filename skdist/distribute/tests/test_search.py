"""
Test search classes
"""

import numpy as np

from scipy.stats.distributions import expon
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

try:
    from skdist.distribute import search
    from skdist.distribute.search import (
        DistGridSearchCV, DistRandomizedSearchCV,
        DistMultiModelSearch
        )
    _import_error = None
except Exception as e:
    _import_error = e

def test_search():
    assert _import_error == None

def test_gs():
    X = np.array([[1,1,1], [0,0,0], [-1,-1,-1]]*100)
    y = np.array([0,0,1]*100)
    gs = DistGridSearchCV(
        LogisticRegression(solver="liblinear"), 
        {"C": [0.1, 1.0]}, cv=3
        )
    gs.fit(X,y)
    preds = gs.predict(X[:3])
    assert np.allclose(preds, np.array([0,0,1]))

def test_rs():
    X = np.array([[1,1,1], [0,0,0], [-1,-1,-1]]*100)
    y = np.array([0,0,1]*100)
    rs = DistRandomizedSearchCV(
        LogisticRegression(solver="liblinear"),
        {"C": [0.1, 1.0]}, cv=3, n_iter=2
        )
    rs.fit(X,y)
    preds = rs.predict(X[:3])
    assert np.allclose(preds, np.array([0,0,1]))

def test_multimodel():
    X = np.array([[1,1,1], [0,0,0], [-1,-1,-1]]*100)
    y = np.array([0,0,1]*100)
    models = [
        ("rf", RandomForestClassifier(n_estimators=10), {"max_depth": [5,10], "min_samples_split": range(2, 10)}),
        ("lr0", LogisticRegression(multi_class="auto", solver="liblinear"), {"C": [0.1, 1.0, 10]}), 
        ("lr1", LogisticRegression(multi_class="auto", solver="liblinear"), {"fit_intercept": [True, False], "C": expon()}), 
        ("nb", GaussianNB(), {})
        ]
    clf = DistMultiModelSearch(models, n=2)
    clf.fit(X,y)
    preds = clf.predict(X[:3])
    assert np.allclose(preds, np.array([0,0,1]))
