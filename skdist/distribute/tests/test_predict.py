"""
Test predict imports
"""

try:
    from skdist.distribute import predict
    from skdist.distribute.predict import get_prediction_udf 
    _import_error = None
except Exception as e:
    _import_error = e

def test_predict():
    assert _import_error == None
