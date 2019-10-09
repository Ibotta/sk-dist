"""
Test postprocessing 
"""

try:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from skdist import postprocessing
    _import_error = None
except Exception as e:
    _import_error = e

def test_import_postprocessing():
    assert _import_error == None

def test_predict():
    X = np.array([[1,2,3], [4,5,6]])
    y = np.array([0,1])

    model1 = LogisticRegression(solver="liblinear")
    model2 = LogisticRegression(solver="lbfgs")
    model1.fit(X,y)
    model2.fit(X,y)
    
    clf = postprocessing.SimpleVoter(
        [("model1", model1), ("model2", model2)],
        voting="soft", classes=model1.classes_
        )
    pred = clf.predict(X)
    probs = clf.predict_proba(X)
    assert pred.shape == y.shape

def test_predict_hard_voting():
    X = np.array([[1,2,3], [4,5,6]])
    y = np.array([0,1])

    model1 = LogisticRegression(solver="liblinear")
    model2 = LogisticRegression(solver="lbfgs")
    model1.fit(X,y)
    model2.fit(X,y)

    clf = postprocessing.SimpleVoter(
        [("model1", model1), ("model2", model2)],
        voting="hard", classes=model1.classes_
        )
    pred = clf.predict(X)
    assert np.allclose(pred, y)

def test_predict_strings():
    X = np.array([[1,2,3], [4,5,6]])
    y = np.array(["pizza","tacos"])

    model1 = LogisticRegression(solver="liblinear")
    model2 = LogisticRegression(solver="lbfgs")
    model1.fit(X,y)
    model2.fit(X,y)

    clf = postprocessing.SimpleVoter(
        [("model1", model1), ("model2", model2)],
        voting="hard", classes=model1.classes_
        )
    pred = clf.predict(X)
    assert list(pred) == list(y)
