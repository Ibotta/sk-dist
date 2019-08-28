"""
Test ensemble classes
"""

try:
    import numpy as np
    from skdist.distribute import ensemble
    from skdist.distribute.ensemble import (
        DistRandomForestClassifier, DistRandomForestRegressor, 
        DistExtraTreesClassifier, DistExtraTreesRegressor,
        DistRandomTreesEmbedding
        )
    _import_error = None
except Exception as e:
    _import_error = e

def test_ensemble():
    assert _import_error == None

def test_rfc():
    X = np.array([[0,1,0,1], [0,0,0,1], [1,0,1,0]])
    y = np.array([0,1,0])
    rfc = ensemble.DistRandomForestClassifier(
        n_estimators=10, random_state=5)
    rfc.fit(X,y)
    preds = rfc.predict(X)
    assert np.allclose(preds, np.array([0,1,0]))
 
def test_rfr():
    X = np.array([[0,1,0,1], [0,0,0,1], [1,0,1,0]])
    y_reg = np.array([0.1, 0.2, 0.1])
    rfr = ensemble.DistRandomForestRegressor(
        n_estimators=10, random_state=5)
    rfr.fit(X,y_reg)
    preds = rfr.predict(X)
    assert np.allclose(preds, np.array([0.15, 0.18, 0.12]))

def test_etc():
    X = np.array([[0,1,0,1], [0,0,0,1], [1,0,1,0]])
    y = np.array([0,1,0])
    etc = ensemble.DistExtraTreesClassifier(
        n_estimators=10, random_state=5)
    etc.fit(X,y)
    preds = etc.predict(X)
    assert np.allclose(preds, np.array([0,1,0]))

def test_etr():
    X = np.array([[0,1,0,1], [0,0,0,1], [1,0,1,0]])
    y_reg = np.array([0.1, 0.2, 0.1])
    etr = ensemble.DistExtraTreesRegressor(
        n_estimators=10, random_state=5)  
    etr.fit(X,y_reg)
    preds = etr.predict(X)
    assert np.allclose(preds, np.array([0.1, 0.2, 0.1]))

def test_rte():
    X = np.array([[0,1,0,1], [0,0,0,1], [1,0,1,0]])
    rte = ensemble.DistRandomTreesEmbedding(
        n_estimators=10, random_state=5)  
    rte.fit(X,y=None)
    preds = rte.transform(X)
    assert preds.shape == (3,30)

