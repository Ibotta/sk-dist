"""
Pyspark unit tests
"""

import pytest
import sys
import pandas as pd
import numpy as np

try:
    import pyspark
    from pyspark.sql import SparkSession, functions as F
except ImportError:
    pass

from sklearn.datasets import (
    load_breast_cancer,
    load_digits
    )
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

try:
    import xgboost
    from xgboost import XGBClassifier
except ImportError:
    pass

from skdist.distribute.multiclass import DistOneVsRestClassifier
from skdist.distribute.search import DistGridSearchCV, DistRandomizedSearchCV
from skdist.distribute.ensemble import DistRandomForestClassifier
from skdist.distribute.predict import get_prediction_udf

@pytest.mark.skipif("pyspark" not in sys.modules, reason="requires pyspark")
def test_spark_session_dataframe(spark_session):
    test_df = spark_session.createDataFrame([[1, 3], [2, 4]], "a: int, b: int")

    assert type(test_df) == pyspark.sql.dataframe.DataFrame
    assert test_df.count() == 2

@pytest.mark.skipif("pyspark" not in sys.modules, reason="requires pyspark")
def test_spark_session_sql(spark_session):
    test_df = spark_session.createDataFrame([[1, 3], [2, 4]], "a: int, b: int")
    test_df.createOrReplaceTempView('test')

    test_filtered_df = spark_session.sql('SELECT a, b from test where a > 1')
    assert test_filtered_df.count() == 1

@pytest.mark.skipif("pyspark" not in sys.modules, reason="requires pyspark")
def test_ensemble(spark_session):
    sc = spark_session.sparkContext

    test_size = 0.2
    max_depth = None
    n_estimators = 100

    # load sample data (binary target)
    data = load_breast_cancer()
    X = data["data"]
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=10
    )

    ### distributed random forest
    model = DistRandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth, sc=sc,
    )
    # distributed fitting with spark
    model.fit(X_train, y_train)
    # predictions on the driver
    preds = model.predict(X_test)

    assert preds.shape == y_test.shape

@pytest.mark.skipif("pyspark" not in sys.modules, reason="requires pyspark")
def test_search(spark_session):
    sc = spark_session.sparkContext

    # sklearn variables
    Cs = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    cv = 5
    test_size = 0.2
    scoring = "roc_auc"
    solver = "liblinear"

    # load sample data (binary target)
    data = load_breast_cancer()
    X = data["data"]
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=10
    )

    ### distributed grid search
    model = DistGridSearchCV(
        LogisticRegression(solver=solver),
        dict(C=Cs), sc, cv=cv, scoring=scoring
    )
    # distributed fitting with spark
    model.fit(X_train, y_train)
    # predictions on the driver
    preds = model.predict(X_test)

    assert preds.shape == y_test.shape

@pytest.mark.skipif("pyspark" not in sys.modules, reason="requires pyspark")
def test_multiclass(spark_session):
    sc = spark_session.sparkContext

    # variables
    solver = "liblinear"
    test_size = 0.2

    # load sample data (binary target)
    data = load_digits()
    X = data["data"]
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=10
    )

    ### distributed one vs rest
    model = DistOneVsRestClassifier(LogisticRegression(solver=solver), sc)
    # distributed fitting with spark
    model.fit(X_train, y_train)
    # predictions on the driver
    preds = model.predict(X_test)

    assert preds.shape == y_test.shape

@pytest.mark.skipif("pyspark" not in sys.modules, reason="requires pyspark")
def test_predict(spark_session):
    sc = spark_session.sparkContext

    # simple 2-D numpy features
    data = load_digits()
    X = data["data"]
    y = data["target"]
    model = LogisticRegression(
        solver="liblinear",
        multi_class="auto"
    )
    model.fit(X, y)

    # get UDFs with default 'numpy' feature types
    predict = get_prediction_udf(model, method="predict")
    predict_proba = get_prediction_udf(model, method="predict_proba")

    # create PySpark DataFrame from features
    pdf = pd.DataFrame(X)
    sdf = spark_session.createDataFrame(pdf)
    cols = [F.col(str(c)) for c in sdf.columns]

    # apply predict UDFs and select prediction output
    prediction_df = (
        sdf
            .withColumn("scores", predict_proba(*cols))
            .withColumn("preds", predict(*cols))
            .select("preds", "scores")
    )
    assert prediction_df.count() == X.shape[0]

@pytest.mark.skipif(("pyspark" not in sys.modules) or ("xgboost" not in sys.modules), reason="requires pyspark and xgboost")
def test_xgboost(spark_session):
    sc = spark_session.sparkContext

    X = np.array([[1,1,1], [0,0,0], [-1,-1,-1]]*100)
    y = np.array([0,0,1]*100)

    clf = DistRandomizedSearchCV(
        XGBClassifier(), {"max_depth": [3,5]},
        cv=3, n_iter=2, sc=sc
        )
    X_test = np.array([[1,1,0], [-2,0,5], [1,1,1]]*10)
    y_test = np.array([1,1,0]*10)
    fit_params = {
        'eval_metric': 'logloss',
        'eval_set': [(X_test, y_test)],
        'early_stopping_rounds': 10
        }
    clf.fit(X, y, **fit_params)
    preds = clf.predict(X[:3])
    assert np.allclose(preds, np.array([0,0,1]))
