"""
Test encoder classes
"""

try:
    import pandas as pd
    import numpy as np
    from skdist.distribute import encoder
    from skdist.distribute.encoder import Encoderizer, EncoderizerExtractor
    _import_error = None
except Exception as e:
    _import_error = e

def test_encoder():
    assert _import_error == None

def test_simple():
    # simple encoderizer example
    X = [
        "text encoding is fun", 
        "this is a text encoding example",
        "hopefully it works and we'll have passed the test"
        ]
    encoderizer = encoder.Encoderizer(col_names=["text"])
    X_t = encoderizer.fit_transform(X)
    assert X_t.shape == (3,31)

def test_pandas():
    # pandas encoderizer example
    X = [
        "text encoding is fun",
        "this is a text encoding example",
        "hopefully it works and we'll have passed the test"
        ]
    df = pd.DataFrame({
        "text": X*4, 
        "constant_int": [np.nan,2,2]*4, 
        "constant_str": ["test","test","test"]*4, 
        "constant_null": [None,"test",np.nan]*4, 
        "numbers": [0.123, None, 0.535]*4,
        "dicts": [{"a": 4}, None, {"b": 1}]*4
        })
    encoderizer = encoder.Encoderizer(size="medium")
    X_t = encoderizer.fit_transform(df)
    assert X_t.shape == (12, 180)

def test_dict_list():
    # pandas encoderizer example
    X = [
        "[text encoding is fun]",
        "this is a text encoding {'pizza': 'example'}",
        "hopefully it works and [we'll have passed] the test"
        ]
    df = pd.DataFrame({
        "text": X*4,
        "constant_str": ["test","test","test"]*4,
        "dicts": [{"a": 4}, None, {"b": 1}]*4,
        "lists": [["this", "is", "text"], ["more", "text"], ["text"]]*4
        })
    encoderizer = encoder.Encoderizer(size="medium")
    X_t = encoderizer.fit_transform(df)
    assert X_t.shape == (12, 210)

def test_dict_list_tuple():
    # pandas encoderizer example
    X = [
        "[text encoding is fun]",
        "this is a text encoding {'pizza': 'example'}",
        "hopefully it works and [we'll have passed] the test"
        ]
    df = pd.DataFrame({
	    "text_col": X*4,
	    "categorical_str_col": ["control", "treatment", "control"]*4,
	    "categorical_int_col": [0, 1, 2]*4,
	    "numeric_col": [5, 22, 69]*4,
	    "dict_col": [{"a": 4}, {"b": 1}, {"c": 3}]*4,
	    "list_col": [[1, 2], [1, 3], [2,]]*4,
	    "tuple_col": [(1, 2), (1, 3), (2,)]*4
	    })
    encoderizer = encoder.Encoderizer(size="medium")
    X_t = encoderizer.fit_transform(df)
    assert X_t.shape == (12, 244)
    