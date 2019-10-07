"""
Base functions for distributed meta-estimators
"""

import copy

def _clone(estimator, safe=True):
    """
    Constructs a new estimator with the same parameters.
    Handles sparkContext which shouldn't be copied.
    """
    found_sc = False
    if hasattr(estimator, "sc"):
        found_sc = True
    estimator_type = type(estimator)
    # XXX: not handling dictionaries
    if estimator_type in (list, tuple, set, frozenset):
        return estimator_type([_clone(e, safe=safe) for e in estimator])
    elif not hasattr(estimator, 'get_params') or isinstance(estimator, type):
        if not safe:
            return copy.deepcopy(estimator)
        else:
            raise TypeError("Cannot clone object '%s' (type %s): "
                            "it does not seem to be a scikit-learn estimator "
                            "as it does not implement a 'get_params' methods."
                            % (repr(estimator), type(estimator)))
    klass = estimator.__class__
    new_object_params = estimator.get_params(deep=False)
    for name, param in new_object_params.items():
        if name != "sc":
            new_object_params[name] = _clone(param, safe=False)
    new_object = klass(**new_object_params)
    params_set = new_object.get_params(deep=False)

    # quick sanity check of the parameters of the clone
    for name in new_object_params:
        param1 = new_object_params[name]
        param2 = params_set[name]
        if param1 is not param2:
            raise RuntimeError('Cannot clone object %s, as the constructor '
                               'either does not set or modifies parameter %s' %
                               (estimator, name))

    if found_sc:
        new_object.sc = estimator.sc
    return new_object

def _parse_partitions(partitions, auto_n):
    """ Handles the partitions input for spark parallelization """
    if partitions is None:
        partitions = None
    elif partitions == 'auto':
        partitions = auto_n
    else:
        try:
            partitions = int(partitions)
        except:
            partitions = None
    return partitions

def _get_value(obj):
    """ 
    Determines if input object is broadcast variable.
    If so return value attribute, else return object.
    """
    return obj.value if hasattr(obj, "value") else obj
