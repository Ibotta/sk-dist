"""
Test search classes
"""

try:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from skdist.distribute import search
    from skdist.distribute.search import (
        DistGridSearchCV, DistRandomizedSearchCV
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

