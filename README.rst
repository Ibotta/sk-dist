.. figure:: https://github.com/Ibotta/sk-dist/blob/master/doc/images/skdist.png
   :alt: sk-dist

sk-dist: Distributed scikit-learn meta-estimators in PySpark
============================================================

|License| |Build Status| |PyPI Package| |Downloads| |Python Versions|

What is it?
-----------

``sk-dist`` is a Python package for machine learning built on top of
`scikit-learn <https://scikit-learn.org/stable/index.html>`__ and is
distributed under the `Apache 2.0 software
license <https://github.com/Ibotta/sk-dist/blob/master/LICENSE>`__. The
``sk-dist`` module can be thought of as "distributed scikit-learn" as
its core functionality is to extend the ``scikit-learn`` built-in
``joblib`` parallelization of meta-estimator training to
`spark <https://spark.apache.org/>`__. A popular use case is the 
parallelization of grid search as shown here:

.. figure:: https://github.com/Ibotta/sk-dist/blob/master/doc/images/grid_search.png
   :alt: sk-dist

Check out the `blog post <https://medium.com/building-ibotta/train-sklearn-100x-faster-bec530fc1f45>`__ 
for more information on the motivation and use cases of ``sk-dist``.

Main Features
-------------

-  **Distributed Training** - ``sk-dist`` parallelizes the training of
   ``scikit-learn`` meta-estimators with PySpark. This allows
   distributed training of these estimators without any constraint on
   the physical resources of any one machine. In all cases, spark
   artifacts are automatically stripped from the fitted estimator. These
   estimators can then be pickled and un-pickled for prediction tasks,
   operating identically at predict time to their ``scikit-learn``
   counterparts. Supported tasks are:

   -  *Grid Search*: `Hyperparameter optimization
      techniques <https://scikit-learn.org/stable/modules/grid_search.html>`__,
      particularly
      `GridSearchCV <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV>`__
      and
      `RandomizedSeachCV <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV>`__,
      are distributed such that each parameter set candidate is trained
      in parallel.
   -  *Multiclass Strategies*: `Multiclass classification
      strategies <https://scikit-learn.org/stable/modules/multiclass.html>`__,
      particularly
      `OneVsRestClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html#sklearn.multiclass.OneVsRestClassifier>`__
      and
      `OneVsOneClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsOneClassifier.html#sklearn.multiclass.OneVsOneClassifier>`__,
      are distributed such that each binary probelm is trained in
      parallel.
   -  *Tree Ensembles*: `Decision tree
      ensembles <https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees>`__
      for classification and regression, particularly
      `RandomForest <https://scikit-learn.org/stable/modules/ensemble.html#random-forests>`__
      and
      `ExtraTrees <https://scikit-learn.org/stable/modules/ensemble.html#extremely-randomized-trees>`__,
      are distributed such that each tree is trained in parallel.

-  **Distributed Prediction** - ``sk-dist`` provides a prediction module
   which builds `vectorized
   UDFs <https://spark.apache.org/docs/latest/sql-pyspark-pandas-with-arrow.html#pandas-udfs-aka-vectorized-udfs>`__
   for
   `PySpark <https://spark.apache.org/docs/latest/api/python/index.html>`__
   `DataFrames <https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame>`__
   using fitted ``scikit-learn`` estimators. This distributes the
   ``predict`` and ``predict_proba`` methods of ``scikit-learn``
   estimators, enabling large scale prediction with ``scikit-learn``.
-  **Feature Encoding** - ``sk-dist`` provides a flexible feature
   encoding utility called ``Encoderizer`` which encodes mix-typed
   feature spaces using either default behavior or user defined
   customizable settings. It is particularly aimed at text features, but
   it additionally handles numeric and dictionary type feature spaces.

Installation
------------

Dependencies
~~~~~~~~~~~~

``sk-dist`` requires:

-  `Python <https://www.python.org/>`__ (>= 3.5)
-  `scikit-learn <https://scikit-learn.org/stable/>`__ (>=0.20.0)
-  `pandas <https://pandas.pydata.org/>`__ (>=0.17.0)
-  `numpy <https://www.numpy.org/>`__ 
-  `scipy <https://www.scipy.org/>`__ 
-  `joblib <https://joblib.readthedocs.io/en/latest/>`__ 

Dependency Notes
~~~~~~~~~~~~~~~~

-  versions of ``numpy``, ``scipy`` and ``joblib`` that are compatible with any supported version of ``scikit-learn`` should be sufficient for ``sk-dist``
- ``sk-dist`` is not supported with Python 2

Spark Dependencies
~~~~~~~~~~~~~~~~~~

Most ``sk-dist`` functionality requires a spark installation as well as
PySpark. Some functionality can run without spark, so spark related
dependencies are not required. The connection between sk-dist and spark
relies solely on a ``sparkContext`` as an argument to various
``sk-dist`` classes upon instantiation.

A variety of spark configurations and setups will work. It is left up to
the user to configure their own spark setup. The testing suite runs
``spark 2.3`` and ``spark 2.4``, though any ``spark 2.0+`` versions 
are expected to work.

Additional spark related dependecies are ``pyarrow``, which is used only
for ``skdist.predict`` functions. This uses vectorized pandas UDFs which
require ``pyarrow>=0.8.0``. Depending on the spark version, it may be
necessary to set
``spark.conf.set("spark.sql.execution.arrow.enabled", "true")`` in the
spark configuration.

User Installation
~~~~~~~~~~~~~~~~~

The easiest way to install ``sk-dist`` is with ``pip``:

::

    pip install --upgrade sk-dist

You can also download the source code:

::

    git clone https://github.com/Ibotta/sk-dist.git

Testing
~~~~~~~

With ``pytest`` installed, you can run tests locally:

::

    pytest sk-dist

Examples
--------

The package contains numerous 
`examples <https://github.com/Ibotta/sk-dist/tree/master/examples>`__ 
on how to use ``sk-dist`` in practice. Examples of note are:

-  `Grid Search with XGBoost <https://github.com/Ibotta/sk-dist/blob/master/examples/search/xgb.py>`__
-  `Spark ML Benchmark Comparison <https://github.com/Ibotta/sk-dist/blob/master/examples/search/spark_ml.py>`__
-  `Encoderizer with 20 Newsgroups <https://github.com/Ibotta/sk-dist/blob/master/examples/encoder/basic_usage.py>`__
-  `One-Vs-Rest vs One-Vs-One <https://github.com/Ibotta/sk-dist/blob/master/examples/multiclass/basic_usage.py>`__
-  `Large Scale Sklearn Prediction with PySpark UDFs <https://github.com/Ibotta/sk-dist/blob/master/examples/predict/basic_usage.py>`_

Gradient Boosting
-----------------

``sk-dist`` has been tested with a number of popular gradient boosting packages that conform to the ``scikit-learn`` API. This 
includes ``xgboost`` and ``catboost``. These will need to be installed in addition to ``sk-dist`` on all nodes of the spark 
cluster via a node bootstrap script. Version compatibility is left up to the user.

Support for ``lightgbm`` is not guaranteed, as it requires `additional installations <https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#linux>`__ on all 
nodes of the spark cluster. This may work given proper installation but has not beed tested with ``sk-dist``.

Background
----------

The project was started at `Ibotta
Inc. <https://medium.com/building-ibotta>`__ on the machine learning
team and open sourced in 2019.

It is currently maintained by the machine learning team at Ibotta. Special
thanks to those who contributed to ``sk-dist`` while it was initially
in development at Ibotta:

-  `Evan Harris <https://github.com/denver1117>`__
-  `Nicole Woytarowicz <https://github.com/nicolele>`__
-  `Mike Lewis <https://github.com/Mikelew88>`__
-  `Bobby Crimi <https://github.com/rpcrimi>`__

.. figure:: https://github.com/Ibotta/sk-dist/blob/master/doc/images/ibottaml.png
   :alt: IbottaML

.. |License| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0
.. |Build Status| image:: https://travis-ci.org/Ibotta/sk-dist.png?branch=master
   :target: https://travis-ci.org/Ibotta/sk-dist
.. |PyPI Package| image:: https://badge.fury.io/py/sk-dist.svg
   :target: https://pypi.org/project/sk-dist/
.. |Downloads| image:: https://img.shields.io/pypi/dm/sk-dist
   :target: https://pypi.org/project/sk-dist/
.. |Python Versions| image:: https://img.shields.io/pypi/pyversions/sk-dist
   :target: https://pypi.org/project/sk-dist/
