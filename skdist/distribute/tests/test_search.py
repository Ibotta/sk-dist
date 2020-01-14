"""
Test search classes
"""

import sys
import pytest
import numpy as np

try:
    import xgboost
    from xgboost import XGBClassifier
except ImportError:
    pass

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

@pytest.mark.skipif("xgboost" not in sys.modules, reason="requires xgboost")
def test_fit_params():
    X = np.array([[1,1,1], [0,0,0], [-1,-1,-1]]*100)
    y = np.array([0,0,1]*100)
    
    clf = DistRandomizedSearchCV(
        XGBClassifier(), {"max_depth": [3,5]}, 
        cv=3, n_iter=2
        )
    X_test = np.array([[1,1,0], [-2,0,5], [1,1,1]]*10) 
    y_test = np.array([1,1,0]*10)
    fit_params = {
        'eval_metric': 'logloss',
        'eval_set': [(X_test, y_test)],
        'early_stopping_rounds': 10
        }
    clf.fit(X, y, **fit_params)    
    preds = clf.predict(X[:3])
    assert np.allclose(preds, np.array([0,0,1]))
