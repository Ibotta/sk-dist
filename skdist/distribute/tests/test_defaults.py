"""
Test defaults imports
"""

try:
    from skdist.distribute import _defaults
    from skdist.distribute._defaults import *
    _import_error = None
except Exception as e:
    _import_error = e

def test_import_defaults():
    assert _import_error == None

def test_tokenizer():
    assert tokenizer(5) == 5

def test_dict():
    func = _defaults.dict_encoder("a")
    assert isinstance(func, list)

def test_numeric():
    func = _defaults.numeric_encoder("a")
    assert isinstance(func, list)

def test_onehot():
    func = _defaults.onehot_encoder("a")
    assert isinstance(func, list)

def test_multihot():
    func = _defaults.multihot_encoder("a")
    assert isinstance(func, list)

def test_small():
    for key, val in _defaults._default_encoders["small"].items():
        func = val("c")
    assert isinstance(func, list)

def test_medium():
    for key, val in _defaults._default_encoders["medium"].items():
        func = val("c")
    assert isinstance(func, list)

def test_large():
    for key, val in _defaults._default_encoders["large"].items():
        func = val("c")
    assert isinstance(func, list)
     
