"""
Test preprocessing
"""

try:
    import pandas as pd
    import numpy as np
    from scipy.sparse import csr_matrix
    from skdist import preprocessing
    _import_error = None
except Exception as e:
    _import_error = e

def test_import_preprocessing():
    assert _import_error == None

def test_varselect():
    vars = ["text"]
    df = pd.DataFrame({"text": ["a", "b", "c"], "numbers": [1, 2, 3]})
    selector = preprocessing.SelectField(cols=vars, single_dimension=True)
    X_t = selector.fit_transform(df)
    assert X_t.shape == (3,)

def test_varselect_multidim():
    vars = ["text"]
    df = pd.DataFrame({"text": ["a", "b", "c"], "numbers": [1, 2, 3]})
    selector = preprocessing.SelectField(cols=vars, single_dimension=False)
    X_t = selector.fit_transform(df)
    assert X_t.shape == (3,1)

def test_featurecast():
    data = np.array([[1,2,3]])
    X_t = preprocessing.FeatureCast(cast_type=float).fit_transform(data)
    assert isinstance(X_t[0][0], np.float64)

def test_fillna():
    X = np.array([["a", np.nan, "c"], [np.nan, 2, 3]])
    selector = preprocessing.ImputeNull("a")
    X_t = selector.fit_transform(X)
    assert X_t.shape == X.shape

def test_densetransformer():
    data = np.array([[1,2,3], [4,5,6]])
    X = csr_matrix(data)
    X_t = preprocessing.DenseTransformer().fit_transform(X)
    assert isinstance(X_t, np.ndarray)

def test_labelencoderpipe():
    X = np.array([1,2,3])
    X_t = preprocessing.LabelEncoderPipe().fit_transform(X)
    assert X_t.shape[0] == X.shape[0]

def test_le_pipe():
    X = np.array([1,2,3])
    X_t = preprocessing.LabelEncoderPipe().fit_transform(X)
    assert X_t.shape != X.shape

def test_selectormem():
    X = np.array([[1,2,3], [4,5,6], [10,10,10]])
    y = np.array([0,0,1])

    X_t = preprocessing.SelectorMem().fit_transform(X,y)
    assert X_t.shape != X.shape

def test_hashingvectorizerchunked():
    X = ["here is some text", "more text here", "this is also text"]
    X_t = preprocessing.HashingVectorizerChunked().fit_transform(X)
    assert X_t.shape == (3,2**20)

def test_hv_chunked():
    X = ["here is some text", "more text here", "this is also text"]
    X_t = preprocessing.HashingVectorizerChunked(chunksize=1).fit_transform(X)
    assert X_t.shape == (3,2**20)

def test_multihot():
    X = [["text", "more text", "this"], ["more text"], ["new text"]]
    X_t = preprocessing.MultihotEncoder().fit_transform(X)
    assert X_t.shape == (3,4)

def test_multihot_sparse():
    X = [["text", "more text", "this"], ["more text"], ["new text"]]
    X_t = preprocessing.MultihotEncoder(sparse_output=True).fit_transform(X)
    assert X_t.shape == (3,4)
  
def test_multihot_unknown():
    X = [["text", "more text", "this"], ["more text"], ["new text"]]
    enc = preprocessing.MultihotEncoder()
    enc.fit(X)
    X_t = enc.transform([["unseen", "this"], ["more text"], ["unknown"]])
    assert X_t.shape == (3,4)
    
