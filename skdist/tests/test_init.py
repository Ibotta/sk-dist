"""
Test top level import
"""

try:
    from skdist import *
    _top_import_error = None
except Exception as e:
    _top_import_error = e

def test_import_skdist():
    assert _top_import_error == None
