"""
Distributed scikit-learn meta-estimators in PySpark
===================================================
skdist is a Python module that aims to efficiently distribute scikit-learn
meta-estimator model fitting with PySpark. Where sklearn uses joblib to
parallelize these operations, distml uses spark, enabling effectively infinite
parallelization, unconstrained by the number of cores on a single machine.

Existing scikit-learn meta-estimators for model selection, ensembling,
multiclass fitting, and feature unioning are inherited, and have their fit methods
overridden with a distributed implementation using spark. The resulting fitted model
is effectively identical to its single machine counterpart,
ready for prediction using the inherited methods.
"""

__version__ = '0.1.9'

__all__ = ['distribute', 'preprocessing', 'postprocessing', 'tests']
