"""
=============================================================
Distribute sklearn model prediction with PySpark using skdist
=============================================================

In this example we train 3 sklearn estimators with 3 different
types of feature spaces. This includes standard 2-D numpy
array ready for direct input into a sklearn estimator, single
field text input ready for input into a text vectorizer piped
into a sklearn estimator, and finally a pandas DataFrame
ready for input into a custom feature pipeline or a sklearn
ColumnTransformer piped into a sklearn estimator.

The skdist.predict module is built to handle all 3 of these
use cases, intended to cover most of the ground needed to 
handle a wide array of prediction use cases.

In these examples, we don't have a large amount of data
at predict time, so we simply parallelize the sample datasets
to a PySpark DataFrame for illustrative purposes. In 
practice, skdist.predict would only be useful if there were
large amounts of data from another source ready for prediction.

Here is a sample output run:

+-----+--------------------+
|preds|              scores|
+-----+--------------------+
|    0|[0.99988026795692...|
|    1|[4.75035277837040...|
|    2|[2.94811218592164...|
|    3|[1.63438595023762...|
|    4|[1.11339868338047...|
|    5|[1.47300432716012...|
|    6|[1.08560009259480...|
|    7|[3.02428232165044...|
|    8|[7.65445972596079...|
|    9|[3.97610488897298...|
|    0|[0.99918670844137...|
|    1|[2.65336456879078...|
|    2|[1.85886361541580...|
|    3|[2.89824009324990...|
|    4|[2.84813979824305...|
|    5|[2.70090567992820...|
|    6|[1.10907772018062...|
|    7|[3.06455862370095...|
|    8|[2.38739344440480...|
|    9|[8.23628591704589...|
+-----+--------------------+
only showing top 20 rows
+-----+--------------------+
|preds|              scores|
+-----+--------------------+
|    4|[0.03736128393565...|
|    0|[0.09792807410478...|
|   17|[0.05044543817914...|
|   11|[0.03443972986074...|
|   10|[0.04757471929521...|
|   15|[0.04555477151025...|
|    4|[0.04025302976824...|
|   17|[0.04606538206124...|
|    4|[0.05296440750891...|
|   12|[0.04526243345294...|
|    4|[0.03733198188990...|
|    6|[0.04041213769366...|
|    4|[0.04252566904405...|
|   15|[0.04738860601686...|
|    4|[0.03942044494467...|
|   11|[0.04281835124858...|
|   11|[0.03675331309090...|
|    4|[0.03287753061778...|
|   12|[0.04517622045917...|
|   11|[0.04878195327579...|
+-----+--------------------+
only showing top 20 rows
+-----+--------------------+
|preds|              scores|
+-----+--------------------+
|    4|[0.03736128393565...|
|    0|[0.09792807410478...|
|   17|[0.05044543817914...|
|   11|[0.03443972986074...|
|   10|[0.04757471929521...|
|   15|[0.04555477151025...|
|    4|[0.04025302976824...|
|   17|[0.04606538206124...|
|    4|[0.05296440750891...|
|   12|[0.04526243345294...|
|    4|[0.03733198188990...|
|    6|[0.04041213769366...|
|    4|[0.04252566904405...|
|   15|[0.04738860601686...|
|    4|[0.03942044494467...|
|   11|[0.04281835124858...|
|   11|[0.03675331309090...|
|    4|[0.03287753061778...|
|   12|[0.04517622045917...|
|   11|[0.04878195327579...|
+-----+--------------------+
only showing top 20 rows
"""
print(__doc__)

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_20newsgroups, load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline
from skdist.distribute.predict import get_prediction_udf
from pyspark.sql import SparkSession, functions as F

# spark session initialization
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

# simple 2-D numpy features
data = load_digits()
X = data["data"]
y = data["target"]
model = LogisticRegression(solver="liblinear", multi_class="auto")
model.fit(X, y)

# get UDFs with default 'numpy' feature types
predict = get_prediction_udf(model, method="predict")
predict_proba = get_prediction_udf(model, method="predict_proba")

# create PySpark DataFrame from features
pdf = pd.DataFrame(X)
sdf = spark.createDataFrame(pdf)
cols = [F.col(str(c)) for c in sdf.columns]

# apply predict UDFs and select prediction output
prediction_df = (
    sdf.withColumn("scores", predict_proba(*cols))
    .withColumn("preds", predict(*cols))
    .select("preds", "scores")
)
prediction_df.show()

# single text feature
data = fetch_20newsgroups(
    shuffle=True, random_state=1, remove=("headers", "footers", "quotes")
)
X = data["data"][:100]
y = data["target"][:100]
model = Pipeline(
    [
        ("vec", HashingVectorizer()),
        ("clf", LogisticRegression(solver="liblinear", multi_class="auto")),
    ]
)
model.fit(X, y)

# get UDFs with 'text' feature types
predict = get_prediction_udf(model, method="predict", feature_type="text")
predict_proba = get_prediction_udf(model, method="predict_proba", feature_type="text")

# create PySpark DataFrame from features
pdf = pd.DataFrame(X)
sdf = spark.createDataFrame(pdf)
cols = [F.col(str(c)) for c in sdf.columns]

# apply predict UDFs and select prediction output
prediction_df = (
    sdf.withColumn("scores", predict_proba(*cols))
    .withColumn("preds", predict(*cols))
    .select("preds", "scores")
)
prediction_df.show()

# complex feature space as pandas DataFrame
X = pd.DataFrame({"text": data["data"][:100]})
y = data["target"][:100]
model = Pipeline(
    [
        ("vec", ColumnTransformer([("text", HashingVectorizer(), "text")])),
        ("clf", LogisticRegression(solver="liblinear", multi_class="auto")),
    ]
)
model.fit(X, y)

# get UDFs with 'pandas' feature types
# NOTE: This time we must supply an ordered list
# of column names to the `get_predict_udf` function
predict = get_prediction_udf(
    model, method="predict", feature_type="pandas", names=list(X.columns)
)
predict_proba = get_prediction_udf(
    model, method="predict_proba", feature_type="pandas", names=list(X.columns)
)

# create PySpark DataFrame from features
sdf = spark.createDataFrame(X)
cols = [F.col(str(c)) for c in sdf.columns]

# apply predict UDFs and select prediction output
prediction_df = (
    sdf.withColumn("scores", predict_proba(*cols))
    .withColumn("preds", predict(*cols))
    .select("preds", "scores")
)
prediction_df.show()
