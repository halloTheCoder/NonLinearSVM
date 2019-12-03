import numpy as np
from itertools import chain

def check_consistent_length(*arrays):
    """Check that all arrays have consistent first dimensions.
    Checks whether all objects in arrays have the same shape or length.
    
    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """

    lengths = [len(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError("Found input variables with inconsistent numbers of"
                         " samples: %r" % [int(l) for l in lengths])

def indexable(*iterables):
    """Make arrays indexable for cross-validation.
    Checks consistent length, passes through None, and ensures that everything
    can be indexed by converting non-interable objects to arrays.
    
    Parameters
    ----------
    *iterables : lists, dataframes, arrays, sparse matrices
        List of objects to ensure sliceability.
    """
    result = []
    for X in iterables:
        if hasattr(X, "__getitem__") or hasattr(X, "iloc"):
            result.append(X)
        elif X is None:
            result.append(X)
        else:
            result.append(np.array(X))
    check_consistent_length(*result)
    return result

def unique_labels(*ys):
    return list(set(chain.from_iterable(np.unique(y) for y in ys)))

def _preprocess_data_labels(y_true, y_pred, labels=None):
    if labels is None:
        labels = unique_labels(y_true, y_pred)
    labels = np.asarray(labels)
    
    return np.asarray(y_true), np.asarray(y_pred), labels