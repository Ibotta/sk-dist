"""
=======================================================
Classify religion articles with sk-dist and SimpleVoter
=======================================================

In this example we classify 20newsgroups articles between
'alt.atheism' and 'talk.religion.misc' using a variety
of text classification pipelines and a final voting
classifier.

The classification pipelines include:
HashingVectorizer(word) -> LogReg
HashingVectorizer(char) -> LogReg
CountVec(word)|CountVec(char) -> KBest Feature Selection -> ERT

The first two pipelines use distributed grid search
to utilize spark parallelization to quickly optimize the
hyperparameters of the text vectorizer and the regression.
The third pipeline uses the distributed ensemble
from sk-dist to train 1000 trees with the extra 
randomized trees (ERT) algorithm.

We see a total fit time of about 4 minutes. From the 
spark logs (not shown here) we see a total task time
of about 107 minutes. Using sk-dist we see about a 
26x performance improvement. This is using a small
spark cluster running 32 cores.

Note the sample output logs. While this is a trivial 
example where the final validation is run on a 
small holdout set (rather than cross validation),
we see how voting classifiers can be valuable. The trees
win on ROC AUC but lose on F1. The character model
wins on F1 but has a lower ROC AUC. The ensembled
voter with constant weights wins on both metrics.

Here is a sample output run:

Word Model Fit Time: 92.20230150222778
Char Model Fit Time: 148.0373730659485
Tree Model Fit Time: 2.3711297512054443
Total Fit Time: 242.61376881599426
-- Word Model --
ROC AUC Score 0.8490936147186148
F1 Score 0.7307692307692308
-- Char Model --
ROC AUC Score 0.8448998917748918
F1 Score 0.7727272727272727
-- Both Model --
ROC AUC Score 0.8373241341991342
F1 Score 0.6797385620915033
-- Model Model --
ROC AUC Score 0.8960362554112554
F1 Score 0.7721518987341771
"""
print(__doc__)

import time

from skdist.distribute.search import DistGridSearchCV
from skdist.distribute.ensemble import DistExtraTreesClassifier
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.datasets import fetch_20newsgroups
from pyspark.sql import SparkSession

# instantiate spark session
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

# load 20newsgroups dataset
categories = ["alt.atheism", "talk.religion.misc"]
dataset = fetch_20newsgroups(
    shuffle=True,
    random_state=1,
    remove=("headers", "footers", "quotes"),
    categories=categories,
)

# parameters, data load and train/test split
scoring = "roc_auc"
cv = 5
X = dataset["data"]
y = dataset["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# define word vector -> regression model
word_pipe = Pipeline(
    steps=[
        ("vec", HashingVectorizer(analyzer="word", decode_error="ignore")),
        ("clf", LogisticRegression()),
    ]
)
word_params = {
    "vec__ngram_range": [(1, 1), (1, 2), (1, 3), (1, 4), (2, 4)],
    "clf__C": [0.1, 1.0, 10.0],
    "clf__solver": ["liblinear", "lbfgs"],
}
word_model = DistGridSearchCV(word_pipe, word_params, sc=sc, cv=cv, scoring=scoring)

# define character vector -> regression model
char_pipe = Pipeline(
    steps=[
        ("vec", HashingVectorizer(analyzer="char_wb", decode_error="ignore")),
        ("clf", LogisticRegression()),
    ]
)
char_params = {
    "vec__ngram_range": [(2, 2), (2, 3), (2, 4), (2, 5), (3, 3), (3, 5)],
    "clf__C": [0.1, 1.0, 10.0],
    "clf__solver": ["liblinear", "lbfgs"],
}
char_model = DistGridSearchCV(char_pipe, char_params, sc=sc, cv=cv, scoring=scoring)

# define word/character vector -> feature selection -> tree ensemble
feature_union = FeatureUnion(
    [
        ("word", CountVectorizer(analyzer="word", decode_error="ignore")),
        ("char", CountVectorizer(analyzer="char_wb", decode_error="ignore")),
    ]
)
both_model = Pipeline(
    steps=[
        ("vec", feature_union),
        ("select", SelectKBest(f_classif, 1000)),
        ("clf", DistExtraTreesClassifier(n_estimators=1000, max_depth=None, sc=sc)),
    ]
)

# fit all models
start = time.time()
word_model.fit(X_train, y_train)
print("Word Model Fit Time: {0}".format(time.time() - start))

start1 = time.time()
char_model.fit(X_train, y_train)
print("Char Model Fit Time: {0}".format(time.time() - start1))

start2 = time.time()
both_model.fit(X_train, y_train)
print("Tree Model Fit Time: {0}".format(time.time() - start2))
print("Total Fit Time: {0}".format(time.time() - start))

# construct voter
model = SimpleVoter(
    [("word", word_model), ("char", char_model), ("both", both_model)],
    classes=word_model.classes_,
    voting="soft",
)

# compute scoring metrics on holdout
for model_tuple in model.estimators + [("model", model)]:
    print("-- {0} Model --".format(model_tuple[0].title()))
    print(
        "ROC AUC Score",
        roc_auc_score(y_test, model_tuple[1].predict_proba(X_test)[:, 1]),
    )
    print("F1 Score", f1_score(y_test, model_tuple[1].predict(X_test)))
