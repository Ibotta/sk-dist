"""
======================================================================
Compare training runtime and model quality between sk-dist and sparkML
======================================================================

In this example we train a logistic regression with grid search
and a random forest on the covertype dataset. The intent is to replicate
the same model training with sk-dist and sparkML to compare both
model performance and run time on the same spark cluster with the same
allocated spark resources.

Note that this is with data of shape = (581012,54) with
a multi-class problem with 7 classes. This is relatively small data 
from the perspective of the allocated spark resources, and medium to 
large data for a single machine scikit-learn problem.

In this example, sk-dist produces higher scoring models with a 
faster runtime than sparkML. While the relative efficiency of 
sparkML vs scikit-learn will vary depending on the shape of the 
data and the algorithm for a single classifier, this example
illustrates how distributing at the meta-estimator level 
(tree ensembles or grid search) using sk-dist can outperform
distirbuting at the training data level with sparkML.

Here is a sample output run:

-- sk-dist LR --
Train Time: 85.67981553077698
Best Model CV Score: 0.7147715077848511
Holdout F1: 0.7117811859568902
-- sk-dist RF --
Train Time: 9.242362976074219
Holdout F1: 0.9536522156386452
-- spark ML LR --
Train Time: 448.40646290779114
Best Model CV Score: 0.5492014106100074
Holdout F1: 0.6979655562575744
-- spark ML RF --
Train Time: 768.528972864151
Holdout F1: 0.8831069298202333
"""
print(__doc__)

import time
import pandas as pd
import numpy as np

from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from skdist.distribute.search import DistGridSearchCV
from skdist.distribute.ensemble import DistRandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression as LR, RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession

# instantiate spark session
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

# load data
data = fetch_covtype()
X = data["data"]
y = data["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# initial scaling
scaler = StandardScaler()
X_train_t = scaler.fit_transform(X_train)
X_test_t = scaler.transform(X_test)

# sk-dist logistic regression w/ grid search
start = time.time()
lr = LogisticRegression(solver="lbfgs", multi_class="auto")
model = DistGridSearchCV(
    lr, {"C": [10.0, 1.0, 0.1, 0.01]}, sc=sc, cv=5, scoring="f1_weighted"
)
model.fit(X_train_t, y_train)
print("-- sk-dist LR --")
print("Train Time: {0}".format(time.time() - start))
print("Best Model CV Score: {0}".format(model.best_score_))
print(
    "Holdout F1: {0}".format(
        f1_score(y_test, model.predict(X_test_t), average="weighted")
    )
)

# sk-dist random forest
start = time.time()
rf = DistRandomForestClassifier(n_estimators=100, max_depth=None, sc=sc)
rf.fit(X_train_t, y_train)
print("-- sk-dist RF --")
print("Train Time: {0}".format(time.time() - start))
print(
    "Holdout F1: {0}".format(f1_score(y_test, rf.predict(X_test_t), average="weighted"))
)

# spark-ify scaled training data
pandas_df = pd.DataFrame(X_train_t)
pandas_df["label"] = y_train
spark_df = spark.createDataFrame(pandas_df)
assembler = VectorAssembler(
    inputCols=[str(a) for a in pandas_df.columns[:-1]], outputCol="features"
)

# spark ML logistic regression w/ grid seach
start = time.time()
lr = LR()
pipeline = Pipeline(stages=[assembler, lr])
paramGrid = ParamGridBuilder().addGrid(lr.regParam, [10.0, 1.0, 0.1, 0.01]).build()
crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=MulticlassClassificationEvaluator(),
    numFolds=5,
    parallelism=8,
)
cvModel = crossval.fit(spark_df)
print("-- spark ML LR --")
print("Train Time: {0}".format(time.time() - start))
print("Best Model CV Score: {0}".format(np.mean(cvModel.avgMetrics)))

# test holdout
pandas_df = pd.DataFrame(X_test_t)
pandas_df["label"] = y_test
eval_df = spark.createDataFrame(pandas_df)
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
print("Holdout F1: {0}".format(evaluator.evaluate(cvModel.transform(spark_df))))

# random forest with spark ML
start = time.time()
rf = RandomForestClassifier(numTrees=100, maxDepth=30)
pipeline = Pipeline(stages=[assembler, rf])
rfModel = pipeline.fit(spark_df)

# test holdout
print("-- spark ML RF --")
print("Train Time: {0}".format(time.time() - start))
print("Holdout F1: {0}".format(evaluator.evaluate(rfModel.transform(eval_df))))
