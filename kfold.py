import random
import numbers
import numpy as np

from utility import check_consistent_length, indexable

class BaseKFold:
    """Base class for KFold and StratifiedKFold"""
    def __init__(self, n_splits=10, shuffle=True): 
        if not isinstance(n_splits, numbers.Integral):
            raise ValueError("The number of folds must be of Integral type. %s of type %s was passed."
                             % (n_splits, type(n_splits)))
        n_splits = int(n_splits)

        if n_splits <= 1:
            raise ValueError("k-fold cross-validation requires at least one train/test split by setting n_splits=2 or more,"
                " got n_splits={0}.".format(n_splits))

        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be True or False; got {0}".format(shuffle))
        
        self.n_splits = n_splits
        self.shuffle = shuffle
        
    def split(self, X, y):
        """Generate indices to split data into training and test set.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.
        
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        indices = np.arange(len(X))
        for test_index in self._iter_test_masks(X, y):
            train_index = indices[np.logical_not(test_index)]
            test_index = indices[test_index]
            yield train_index, test_index
    
    def _iter_test_masks(self, X, y):
        """Generates boolean masks corresponding to test sets.
        By default, delegates to _iter_test_indices(X, y)
        """
        raise NotImplementedError
    
    def get_n_splits(self):
        return self.n_splits


class KFold(BaseKFold):
    """
    K-Folds cross-validator
    Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds (without shuffling by default).
    Each fold is then used once as a validation while the k - 1 remaining folds form the training set.

    Parameters
    ----------
    n_splits : int, default=10
               Number of folds. Must be at least 2.

    shuffle : boolean, optional
              Whether to shuffle the data before splitting into batches.
    """
    def __init__(self, n_splits=10, shuffle=True): 
        super().__init__(n_splits, shuffle)
    
    def split(self, X, y):
        """Generate indices to split data into training and test set.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.
        
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        for train, test in super().split(X, y):
            yield train, test
        
    def _iter_test_masks(self, X, y):
        """Generates boolean masks corresponding to test sets.
        By default, delegates to _iter_test_indices(X, y)
        """
        for test_index in self._iter_test_indices(X, y):
            test_mask = np.zeros(len(X), dtype=np.bool)
            test_mask[test_index] = True
            yield test_mask
            
    def _iter_test_indices(self, X, y):
        """Generates integer indices corresponding to test sets."""
        n_samples = len(X)
        indices = np.arange(n_samples)
        if self.shuffle:
            random.shuffle(indices)

        n_splits = self.n_splits
        fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=np.int)
        fold_sizes[:n_samples % n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield indices[start:stop]
            current = stop


class StratifiedKFold(BaseKFold):
    """Stratified K-Folds cross-validator
    Provides train/test indices to split data in train/test sets.
    This cross-validation object is a variation of KFold that returns
    stratified folds. The folds are made by preserving the percentage of
    samples for each class.
    
    Parameters
    ----------
    n_splits : int, default=10
        Number of folds. Must be at least 2.
    shuffle : boolean, optional
        Whether to shuffle each class's samples before splitting into batches.
    """
    def __init__(self, n_splits=10, shuffle=True):
        super().__init__(n_splits, shuffle)
        
    def split(self, X, y):
        """Generate indices to split data into training and test set.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.
        
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        for train, test in super().split(X, y):
            yield train, test
            
    def _iter_test_masks(self, X, y):
        """Generates boolean masks corresponding to test sets.
        By default, delegates to _iter_test_indices(X, y)
        """
        test_folds = self._make_test_folds(X, y)
        for i in range(self.n_splits):
            yield test_folds == i
            
    def _make_test_folds(self, X, y):
        _, y_idx, y_inv = np.unique(y, return_index=True, return_inverse=True)
        _, class_perm = np.unique(y_idx, return_inverse=True)
        y_encoded = class_perm[y_inv]
        
        n_classes = len(y_idx)
        y_counts = np.bincount(y_encoded)
        min_groups = np.min(y_counts)
        
        y_order = np.sort(y_encoded)
        allocation = np.asarray([np.bincount(y_order[i::self.n_splits], minlength=n_classes)
                                 for i in range(self.n_splits)])
        
        test_folds = np.empty(len(y), dtype='i')
        for k in range(n_classes):
            folds_for_class = np.arange(self.n_splits).repeat(allocation[:, k])
            if self.shuffle:
                random.shuffle(folds_for_class)
            test_folds[y_encoded == k] = folds_for_class
        return test_folds
