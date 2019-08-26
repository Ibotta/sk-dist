"""
Test base imports
"""

try:
    from skdist.distribute import base
    from skdist.distribute.base import *
    from skdist.distribute.ensemble import DistRandomForestClassifier
    _import_error = None
except Exception as e:
    _import_error = e

def test_import_base():
    assert _import_error == None

def test_clone():
    rf = DistRandomForestClassifier(n_estimators=10)
    rf_cloned = base._clone(rf)
    assert rf.n_estimators == rf_cloned.n_estimators
