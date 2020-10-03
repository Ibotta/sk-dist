"""
======================================
Pipeline example with DistGridSearchCV
======================================

Here we use DistGridSearchCV in sklearn Pipelines in
two different ways. The first packages up a standard
sklearn Pipeline and uses it as the base estimator
in a DistGridSearchCV hyperparameter tuning job.
The second uses DistGridSearchCV as the final estimator
in a standard sklearn Pipeline.

Here is a sample output run:
A Pipeline used as the base estimator for DistGridSearchCV: 0.5974610578071026
DistGridSearchCV at the end of a Pipeline: 0.5110438839310488
"""
print(__doc__)

from skdist.distribute.search import DistGridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from pyspark.sql import SparkSession

# load 20newsgroups dataset
dataset = fetch_20newsgroups(
    shuffle=True, random_state=1, remove=("headers", "footers", "quotes")
)

# variables
cv = 10
scoring = "f1_weighted"
X = dataset["data"]
y = dataset["target"]

# instantiate a pipeline and grid
pipe = Pipeline(
    steps=[
        ("vec", TfidfVectorizer(decode_error="ignore", analyzer="word")),
        ("svd", TruncatedSVD()),
        ("clf", LogisticRegression(solver="liblinear", multi_class="auto")),
    ]
)
params = {
    "clf__C": [0.1, 1.0, 10.0],
    "vec__ngram_range": [(1, 1), (1, 2)],
    "svd__n_components": [50, 100],
}

# fit and select hyperparameters with skdist
model0 = DistGridSearchCV(pipe, params, sc, scoring=scoring, cv=cv)
model0.fit(X, y)
print(
    "A Pipeline used as the base estimator for DistGridSearchCV: {0}".format(
        model0.best_score_
    )
)

# assemble a pipeline with skdist distributed
# grid search as the final estimator step
model1 = Pipeline(
    steps=[
        ("vec", TfidfVectorizer(decode_error="ignore", analyzer="word")),
        ("svd", TruncatedSVD(n_components=50)),
        (
            "clf",
            DistGridSearchCV(
                LogisticRegression(solver="liblinear", multi_class="auto"),
                {"C": [0.1, 1.0, 10.0]},
                sc,
                scoring=scoring,
                cv=cv,
            ),
        ),
    ]
)
model1.fit(X, y)
print(
    "DistGridSearchCV at the end of a Pipeline: {0}".format(
        model1.steps[-1][1].best_score_
    )
)
