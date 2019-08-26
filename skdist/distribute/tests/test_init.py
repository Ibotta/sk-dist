"""
Test top level import
"""

try:
    from skdist.distribute import *
    _top_import_error = None
except Exception as e:
    _top_import_error = e

def test_import_skdist_distribute():
    assert _top_import_error == None
