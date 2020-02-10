"""
Functions for generating vectorized spark UDFs
to distribute the prediction of sklearn model predictions
with PySpark DataFrames
"""

import pandas as pd
import numpy as np

class PysparkRequired(ImportError):
    pass

class PyarrowRequired(ImportError):
    pass

_PYSPARK_INSTALLED = None
_PYARROW_INSTALLED = None

def _is_pyspark_installed():
    global _PYSPARK_INSTALLED
    if _PYSPARK_INSTALLED is None:
        try:
            import pyspark
            _PYSPARK_INSTALLED = True
        except ImportError:
            _PYSPARK_INSTALLED = False

    if _PYSPARK_INSTALLED:
        import pyspark
        return True
    else:
        return False

def _is_pyarrow_installed():
    global _PYARROW_INSTALLED
    if _PYARROW_INSTALLED is None:
        try:
            import pyarrow
            _PYARROW_INSTALLED = True
        except ImportError:
            _PYARROW_INSTALLED = False

    if _PYARROW_INSTALLED:
        import pyarrow
        return True
    else:
        return False

def _get_vals(*cols, feature_type="numpy", names=None):
    """ Prep input data for prediction method """
    if feature_type == "numpy":
        vals = np.transpose([a.values for a in cols])
    elif feature_type == "pandas":
        if names is None:
            raise Exception("Must pass names argument for pandas feature type")
        vals = pd.DataFrame(np.transpose([a.values for a in cols]), columns=names)
    elif feature_type == "text":
        vals = cols[0].values
    else:
        raise ValueError("Unknown feature_type: {0}".format(feature_type))
    return vals

def get_prediction_udf(model, method="predict", feature_type="numpy", names=None):
    """
    Build a vectorized PySpark UDF to apply a sklearn model's `predict` or 
    `predict_proba` methods columns in a PySpark DataFrame. Handles
    flexible types of feature data for prediction including 2-D numpy
    arrays ('numpy'), single field text data ('text') and mixed type
    pandas DataFrames ('pandas'). The UDF can then be applied as shown in the 
    example below.

    NOTE: This function requires pyarrow and pyspark with appropriate
    versions for vectorized pandas UDFs and appropriate spark configuration
    to use pyarrow. Ths requires pyarrow>=0.8.0 and pyspark>=2.3.0. 
    Additionally, the spark version must be 2.3 or higher. These requirements 
    are not enforced by the sk-dist package at setup time.

    Args:
        model (sklearn Estimator): sklearn model to distribute
            predictions with PySpark
        method (str): name of prediction method; either 'predict'
            or 'predict_proba'
        feature_type (str): name of feature type; either 'numpy',
            'pandas' or 'text'
        names (array-like): list of ordered column names
            (only necessary for 'pandas' feature_type
    Returns:
        PySpark pandas UDF (pyspark.sql.functions.pandas_udf)    
    Example:
    >>> import pandas as pd
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.linear_model import LogisticRegression
    >>> from pyspark.sql import SparkSession
    >>> spark = (
    >>>     SparkSession
    >>>     .builder
    >>>     .getOrCreate()
    >>>     )
    >>> data = load_digits()
    >>> X = data["data"]
    >>> y = data["target"]
    >>> model = LogisticRegression()
    >>> model.fit(X, y)
    >>> predict = get_prediction_udf(model, method="predict")
    >>> predict_proba = get_prediction_udf(model, method="predict_proba")
    >>> pdf = pd.DataFrame(X)
    >>> sdf = spark.createDataFrame(pdf)
    >>> cols = [F.col(str(c)) for c in sdf.columns]
    >>> prediction_df = (
    >>>     sdf
    >>>     .withColumn("scores", predict_proba(*cols))
    >>>     .withColumn("preds", predict(*cols))
    >>>     .select("preds", "scores")
    >>>     )
    >>> prediction_df.show() 
    ... +-----+--------------------+
    ... |preds|              scores|
    ... +-----+--------------------+
    ... |    0|[0.99988026795692...|
    ... |    1|[4.75035277837040...|
    ... |    2|[2.94811218592164...|
    ... |    3|[1.63438595023762...|
    ... |    4|[1.11339868338047...|
    ... |    5|[1.47300432716012...|
    ... |    6|[1.08560009259480...|
    ... |    7|[3.02428232165044...|
    ... |    8|[7.65445972596079...|
    ... |    9|[3.97610488897298...|
    ... |    0|[0.99918670844137...|
    ... |    1|[2.65336456879078...|
    ... |    2|[1.85886361541580...|
    ... |    3|[2.89824009324990...|
    ... |    4|[2.84813979824305...|
    ... |    5|[2.70090567992820...|
    ... |    6|[1.10907772018062...|
    ... |    7|[3.06455862370095...|
    ... |    8|[2.38739344440480...|
    ... |    9|[8.23628591704589...|
    ... +-----+--------------------+
    ... only showing top 20 rows   
    """
    if not _is_pyspark_installed():
        raise ImportError("Module pyspark not found")
    if not _is_pyarrow_installed():
        raise ImportError("Module pyarrow not found")
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        DoubleType, StringType, IntegerType, ArrayType
        )

    if method == "predict":
        def predict_func(*cols):
            vals = _get_vals(
                *cols, feature_type=feature_type, 
                names=names
                )
            return pd.Series(model.predict(vals))
        return_type = (
            StringType() 
            if isinstance(model.classes_[0], str) 
            else IntegerType()
            )
        predict = F.pandas_udf(predict_func, returnType=return_type)
    elif method == "predict_proba":  
        def predict_func(*cols):
            vals = _get_vals(
                *cols, feature_type=feature_type, 
                names=names
                )
            return pd.Series(list(model.predict_proba(vals)))
        predict = F.pandas_udf(predict_func, returnType=ArrayType(DoubleType()))
    else:
        raise ValueError("Unknown method: {0}".format(method))
    return predict
