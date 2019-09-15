from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

import pyspark
from skdist.distribute.ensemble import DistRandomForestClassifier

def test_spark_session_dataframe(spark_session):
    test_df = spark_session.createDataFrame([[1, 3], [2, 4]], "a: int, b: int")

    assert type(test_df) == pyspark.sql.dataframe.DataFrame
    assert test_df.count() == 2


def test_spark_session_sql(spark_session):
    test_df = spark_session.createDataFrame([[1, 3], [2, 4]], "a: int, b: int")
    test_df.registerTempTable('test')

    test_filtered_df = spark_session.sql('SELECT a, b from test where a > 1')
    assert test_filtered_df.count() == 1


def test_dist_fit_predict(spark_session):
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