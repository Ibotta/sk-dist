"""
Validation functions for the distribute module
"""

def _check_estimator(estimator, verbose=False):
    """ Print sparkContext awareness if apporpriate """
    if verbose:
        if estimator.sc is None:
            print("No spark context is provided; running locally")
        else:
            print("Spark context found; running with spark")
