"""
Test multiclass classes
"""

try:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from skdist.distribute import multiclass
    from skdist.distribute.multiclass import (
        DistOneVsRestClassifier, DistOneVsOneClassifier  
        )
    _import_error = None
except Exception as e:
    _import_error = e

def test_multiclass():
    assert _import_error == None

def test_ovr():
    X = np.array([[0,0,1,1], [1,1,0,0], [-1,-1,-1,-1]]*100)
    y = np.array([0,1,2]*100)
    ovr = DistOneVsRestClassifier(LogisticRegression(solver="liblinear"))
    ovr.fit(X,y)
    preds = ovr.predict(X[:3])
    assert np.allclose(preds, np.array([0,1,2]))

def test_ovo():
    X = np.array([[0,0,1,1], [1,1,0,0], [-1,-1,-1,-1]]*100)
    y = np.array([0,1,2]*100)
    ovo = DistOneVsOneClassifier(LogisticRegression(solver="liblinear"))
    ovo.fit(X,y)
    preds = ovo.predict(X[:3])
    assert np.allclose(preds, np.array([0,1,2]))
