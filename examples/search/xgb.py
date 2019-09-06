import pickle

from skdist.distribute.search import DistGridSearchCV
from sklearn.datasets import (
  load_breast_cancer,
  load_boston
)
from xgboost import (
  XGBClassifier,
  XGBRegressor
)
from pyspark.sql import SparkSession

# spark session initialization
spark = (
    SparkSession
    .builder
    .getOrCreate()
    )
sc = spark.sparkContext

### XGBClassifier ###

# load sample data (binary target)
data = load_breast_cancer()
X = data["data"]
y = data["target"]

grid = dict(
    learning_rate=[.05, .01],
    max_depth=[4, 6, 8],
  	colsample_bytree=[.6, .8, 1.0],
    n_estimators=[100, 200, 300]
)

### distributed grid search
model = DistGridSearchCV(
    XGBClassifier(),
    grid, sc, cv=5, scoring="roc_auc"
    )
# distributed fitting with spark
model.fit(X,y)
# predictions on the driver
preds = model.predict(X)
probs = model.predict_proba(X)

# results
print("-- Grid Search --")
print("Best Score: {0}".format(model.best_score_))
print("Best colsample_bytree: {0}".format(model.best_estimator_.colsample_bytree))
print("Best learning_rate: {0}".format(model.best_estimator_.learning_rate))
print("Best max_depth: {0}".format(model.best_estimator_.max_depth))
print("Best n_estimators: {0}".format(model.best_estimator_.n_estimators))
print(pickle.loads(pickle.dumps(model)))

### XGBRegressor ###

# load sample data (continuous target)
data = load_boston()
X = data["data"]
y = data["target"]

grid = dict(
    learning_rate=[.05, .01],
    max_depth=[4, 6, 8],
  	colsample_bytree=[.6, .8, 1.0],
    n_estimators=[100, 200, 300]
)

### distributed grid search
model = DistGridSearchCV(
    XGBRegressor(objective='reg:squarederror'),
    grid, sc, cv=5, scoring="neg_mean_squared_error"
    )
# distributed fitting with spark
model.fit(X,y)
# predictions on the driver
preds = model.predict(X)

# results
print("-- Grid Search --")
print("Best Score: {0}".format(model.best_score_))
print("Best colsample_bytree: {0}".format(model.best_estimator_.colsample_bytree))
print("Best learning_rate: {0}".format(model.best_estimator_.learning_rate))
print("Best max_depth: {0}".format(model.best_estimator_.max_depth))
print("Best n_estimators: {0}".format(model.best_estimator_.n_estimators))
print(pickle.loads(pickle.dumps(model)))