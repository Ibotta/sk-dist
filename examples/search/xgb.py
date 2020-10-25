"""
====================================================================================
Distribute hyperparameter tuning with gradient boosting trees via DistGridSearchCV
====================================================================================

In this example we train a classifier and regression with XGBoost by distributing
the hyperparameter tuning through DistGridSearchCV. This should work right out of the
box with XGBoost's sklearn wrapper.

Given the sequential nature of training estimators on gradient boosting trees, it
makes sense to distribute the hyperparameters and cross validation folds, rather than
trying to train multiple estimators in parallel. Skdist excels in this functionality by
leveraging DistGridSearchCV. In this example, we are able to train 54 unique sets of
hyperparameters in parallel and return the the best model to the driver.

NOTE: This example uses xgboost==0.90

Here is a sample output run:

-- Grid Search --
Best Score: 0.9936882800963308
Best colsample_bytree: 1.0
Best learning_rate: 0.05
Best max_depth: 4
Best n_estimators: 300
DistGridSearchCV(cv=5, error_score='raise-deprecating',
                 estimator=XGBClassifier(base_score=0.5, booster='gbtree',
                                         colsample_bylevel=1,
                                         colsample_bynode=1, colsample_bytree=1,
                                         gamma=0, learning_rate=0.1,
                                         max_delta_step=0, max_depth=3,
                                         min_child_weight=1, missing=nan,
                                         n_estimators=100, n_jobs=1,
                                         nthread=None,
                                         objective='binary:logistic',
                                         random_state=0, reg_alpha=0,
                                         reg_lambda=1, scale_pos_weight=1,
                                         seed=None, silent=None, subsample=1,
                                         verbosity=1),
                 iid='warn', n_jobs=None,
                 param_grid={'colsample_bytree': [0.6, 0.8, 1.0],
                             'learning_rate': [0.05, 0.01],
                             'max_depth': [4, 6, 8],
                             'n_estimators': [100, 200, 300]},
                 partitions='auto', pre_dispatch='2*n_jobs', preds=False,
                 refit=True, return_train_score=False, sc=None,
                 scoring='roc_auc', verbose=0)
-- Grid Search --
Best Score: -18.452273211144295
Best colsample_bytree: 0.8
Best learning_rate: 0.05
Best max_depth: 4
Best n_estimators: 200
DistGridSearchCV(cv=5, error_score='raise-deprecating',
                 estimator=XGBRegressor(base_score=0.5, booster='gbtree',
                                        colsample_bylevel=1, colsample_bynode=1,
                                        colsample_bytree=1, gamma=0,
                                        importance_type='gain',
                                        learning_rate=0.1, max_delta_step=0,
                                        max_depth=3, min_child_weight=1,
                                        missing=nan, n_estimators=100, n_jobs=1,
                                        nthread=None,
                                        objective='reg:squarederror',
                                        random...
                                        reg_lambda=1, scale_pos_weight=1,
                                        seed=None, silent=None, subsample=1,
                                        verbosity=1),
                 iid='warn', n_jobs=None,
                 param_grid={'colsample_bytree': [0.6, 0.8, 1.0],
                             'learning_rate': [0.05, 0.01],
                             'max_depth': [4, 6, 8],
                             'n_estimators': [100, 200, 300]},
                 partitions='auto', pre_dispatch='2*n_jobs', preds=False,
                 refit=True, return_train_score=False, sc=None,
                 scoring='neg_mean_squared_error', verbose=0)
"""
print(__doc__)

import pickle

from skdist.distribute.search import DistGridSearchCV
from sklearn.datasets import load_breast_cancer, load_boston
from xgboost import XGBClassifier, XGBRegressor
from pyspark.sql import SparkSession

# spark session initialization
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

### XGBClassifier ###

# sklearn variables
cv = 5
clf_scoring = "roc_auc"
reg_scoring = "neg_mean_squared_error"

# load sample data (binary target)
data = load_breast_cancer()
X = data["data"]
y = data["target"]

grid = dict(
    learning_rate=[0.05, 0.01],
    max_depth=[4, 6, 8],
    colsample_bytree=[0.6, 0.8, 1.0],
    n_estimators=[100, 200, 300],
)

### distributed grid search
model = DistGridSearchCV(XGBClassifier(), grid, sc, cv=cv, scoring=clf_scoring)
# distributed fitting with spark
model.fit(X, y)
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
    learning_rate=[0.05, 0.01],
    max_depth=[4, 6, 8],
    colsample_bytree=[0.6, 0.8, 1.0],
    n_estimators=[100, 200, 300],
)

### distributed grid search
model = DistGridSearchCV(
    XGBRegressor(objective="reg:squarederror"), grid, sc, cv=cv, scoring=reg_scoring
)
# distributed fitting with spark
model.fit(X, y)
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
