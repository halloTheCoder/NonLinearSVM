import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix

from utility import unique_labels, _preprocess_data_labels, stable_cumsum

def confusion_matrix(y_true, y_pred, labels=None, sample_weight=None, normalize=None):
    """
    Compute confusion matrix for the classification.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.
    labels : array-like of shape (n_classes), default=None
        List of labels to index the matrix. This may be used to reorder
        or select a subset of labels.
        If ``None`` is given, those that appear at least once
        in ``y_true`` or ``y_pred`` are used in sorted order.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    normalize : {'true', 'pred', 'all'}, default=None
        Normalizes confusion matrix over the true (rows), predicted (columns)
        conditions or all the population. If None, confusion matrix will not be
        normalized.
    Returns
    -------
    C : ndarray of shape (n_classes, n_classes)
        Confusion matrix.
    """
    y_true, y_pred, labels = _preprocess_data_labels(y_true, y_pred, labels)
    
    if not sample_weight:
        sample_weight = np.ones(y_true.shape[0], dtype=np.int64)
    sample_weight = np.asarray(sample_weight)
    
    n_labels = labels.size
    label_to_ind = {y: x for x, y in enumerate(labels)}
    
    # convert yt, yp into index
    y_pred = np.array([label_to_ind.get(x, n_labels + 1) for x in y_pred])
    y_true = np.array([label_to_ind.get(x, n_labels + 1) for x in y_true])

    # intersect y_pred, y_true with labels, eliminate items not in labels
    ind = np.logical_and(y_pred < n_labels, y_true < n_labels)
    y_pred = y_pred[ind]
    y_true = y_true[ind]
    # also eliminate weights of eliminated items
    sample_weight = sample_weight[ind]
    
    cm = coo_matrix((sample_weight, (y_true, y_pred)), shape=(n_labels, n_labels)).toarray()

    with np.errstate(all='ignore'):
        if normalize == 'true':
            cm = cm / cm.sum(axis=1, keepdims=True)
        elif normalize == 'pred':
            cm = cm / cm.sum(axis=0, keepdims=True)
        elif normalize == 'all':
            cm = cm / cm.sum()
        cm = np.nan_to_num(cm)
    
    return cm

def accuracy_score(y_true, y_pred, labels=None, sample_weight=None, normalize=True):
    """
    Compute accuracy score for the classification.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.
    labels : array-like of shape (n_classes,) default=None
        List of labels to index the matrix. This may be used to reorder
        or select a subset of labels.
        If ``None`` is given, those that appear at least once
        in ``y_true`` or ``y_pred`` are used in sorted order.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    normalize : bool, optional (default=True)
        If ``False``, return the number of correctly classified samples.
        Otherwise, return the fraction of correctly classified samples.
    
    Returns
    -------
    score : float
        If ``normalize == True``, return the fraction of correctly
        classified samples (float), else returns the number of correctly classified samples (int).
        The best performance is 1 with ``normalize == True`` and the number
        of samples with ``normalize == False``
    """
    y_true, y_pred, labels = _preprocess_data_labels(y_true, y_pred, labels)
    
    score = (y_true == y_pred)
    
    if normalize:
        return np.average(score, weights=sample_weight)
    elif sample_weight is not None:
        return np.dot(score, sample_weight)
    else:
        return sample_score.sum()

def f1_score(y_true, y_pred, labels=None, average=None, sample_weight=None):
    """Compute the F1 score for the classification.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.
    labels : array-like of shape (n_classes,) default=None
        List of labels to index the matrix. This may be used to reorder
        or select a subset of labels.
        If ``None`` is given, those that appear at least once
        in ``y_true`` or ``y_pred`` are used in sorted order.
    average : string, [None (default), 'micro', 'macro', 'weighted']
        This parameter is required for multiclass targets.
        ``None`` : 
        the scores for each class are returned
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
        
    Returns
    -------
    f1_score : float
        F1 score of the positive class in binary classification or weighted
        average of the F1 scores of each class for the multiclass task.
    """
    y_true, y_pred, labels = _preprocess_data_labels(y_true, y_pred, labels)
    
    CM = confusion_matrix(y_true, y_pred, labels=labels, sample_weight=sample_weight)
    n_labels = labels.size
    
    with np.errstate(all='ignore'):
        precision_per_class = np.nan_to_num(np.asarray([CM[i, i] / CM[:, i].sum() for i in range(n_labels)]))
        recall_per_class = np.nan_to_num(np.asarray([CM[i, i] / CM[i, :].sum() for i in range(n_labels)]))
        f1_score_per_class = np.nan_to_num(2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class))
        support = CM.sum(axis=1)

    if not average:
        return precision_per_class, recall_per_class, f1_score_per_class, support
    
    if average == 'micro':
        micro_precision = micro_recall = micro_f1_score = accuracy_score(y_true, y_pred)
        return micro_precision, micro_recall, micro_f1_score, support
        
    else:
        if average == 'macro':
            weights = np.ones(n_labels, dtype=np.int64)/n_labels
        elif average == 'weighted':
            weights = support / CM.sum()
        
        macro_precision = np.dot(weights, precision_per_class)
        macro_recall = np.dot(weights, recall_per_class)
        macro_f1_score = np.dot(weights, f1_score_per_class)
        
        return macro_precision, macro_recall, macro_f1_score, support

def classification_report(y_true, y_pred, labels=None, average=None, sample_weight=None):
    """Build a text report showing the main classification metrics.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.
    labels : array-like of shape (n_classes,) default=None
        List of labels to index the matrix. This may be used to reorder
        or select a subset of labels.
        If ``None`` is given, those that appear at least once
        in ``y_true`` or ``y_pred`` are used in sorted order.
    average : string, [None (default), 'micro', 'macro', 'weighted']
        This parameter is required for multiclass targets.
        ``None`` : 
        the scores for each class are returned
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
        
    Returns
    -------
    f1_score : float
        F1 score of the positive class in binary classification or weighted
        average of the F1 scores of each class for the multiclass task.
    """
    y_true, y_pred, labels = _preprocess_data_labels(y_true, y_pred, labels)
    n_labels = len(labels)
    p, r, f1, s = f1_score(y_true, y_pred, labels=labels, average=None, sample_weight=sample_weight)
    
    headers = ["precision", "recall", "f1-score", "support"]
    target_names = ['%s' % l for l in labels]
    rows = zip(target_names, p, r, f1, s)
    average_options = ('micro', 'macro', 'weighted')
    
    longest_last_line_heading = 'weighted avg'
    name_width = max(len(cn) for cn in target_names)
    width = max(name_width, len(longest_last_line_heading), 2)
    head_fmt = '{:>{width}s} ' + ' {:>9}' * len(headers)
    report = head_fmt.format('', *headers, width=width)
    report += '\n\n'
    row_fmt = '{:>{width}s} ' + ' {:>9.{digits}f}' * 3 + ' {:>9}\n'
    for row in rows:
        report += row_fmt.format(*row, width=width, digits=2)
    report += '\n'
    
    # compute all average options
    for average in average_options:
        if average.startswith('micro'):
            line_heading = 'accuracy'
        else:
            line_heading = average + ' avg'
        
        # compute averages with specified averaging method
        avg_p, avg_r, avg_f1, _ = f1_score(y_true, y_pred, labels=labels, average=average, sample_weight=sample_weight)
        avg = [avg_p, avg_r, avg_f1, np.sum(s)]
        
        if line_heading == 'accuracy':
            row_fmt_accuracy = '{:>{width}s} ' + ' {:>9.{digits}}' * 2 + ' {:>9.{digits}f} {:>9}\n'
            report += row_fmt_accuracy.format(line_heading, '', '', *avg[2:], width=width, digits=2)
        else:
            report += row_fmt.format(line_heading, *avg,
                                     width=width, digits=2)
    
    return report

def _binary_clf_curve(y_true, y_score, pos_label=None, sample_weight=None):
    """Calculate true and false positives per binary classification threshold.
    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True targets of binary classification
    y_score : array, shape = [n_samples]
        Estimated probabilities or decision function
    pos_label : int or str, default=None
        The label of the positive class
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    Returns
    -------
    fps : array, shape = [n_thresholds]
        A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds[i]. The total number of
        negative samples is equal to fps[-1] (thus true negatives are given by
        fps[-1] - fps).
    tps : array, shape = [n_thresholds <= len(np.unique(y_score))]
        An increasing count of true positives, at index i being the number
        of positive samples assigned a score >= thresholds[i]. The total
        number of positive samples is equal to tps[-1] (thus false negatives
        are given by tps[-1] - tps).
    thresholds : array, shape = [n_thresholds]
        Decreasing score values.
    """
    y_true, y_pred, labels = _preprocess_data_labels(y_true, y_pred)
    
    if pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)
    
    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.
    
    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true * weight)[threshold_idxs]
    if sample_weight is not None:
        # express fps as a cumsum to ensure fps is increasing even in
        # the presence of floating point errors
        fps = stable_cumsum((1 - y_true) * weight)[threshold_idxs]
    else:
        fps = 1 + threshold_idxs - tps
    return fps, tps, y_score[threshold_idxs]
    
    
def roc_curve(y_true, y_score, pos_label=None, sample_weight=None):
    """Compute Receiver operating characteristic (ROC)
    Note: this implementation is restricted to the binary classification task.
    
    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True binary labels. If labels are not either {-1, 1} or {0, 1}, then
        pos_label should be explicitly given.
    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).
    pos_label : int or str, default=None
        The label of the positive class.
        When ``pos_label=None``, if y_true is in {-1, 1} or {0, 1},
        ``pos_label`` is set to 1, otherwise an error will be raised.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    
    Returns
    -------
    fpr : array, shape = [>2]
        Increasing false positive rates such that element i is the false
        positive rate of predictions with score >= thresholds[i].
    tpr : array, shape = [>2]
        Increasing true positive rates such that element i is the true
        positive rate of predictions with score >= thresholds[i].
    thresholds : array, shape = [n_thresholds]
        Decreasing thresholds on the decision function used to compute
        fpr and tpr. `thresholds[0]` represents no instances being predicted
        and is arbitrarily set to `max(y_score) + 1`.
    """
    fps, tps, thresholds = _binary_clf_curve(y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)
    
    # Add an extra threshold position
    # to make sure that the curve starts at (0, 0)
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[thresholds[0] + 1, thresholds]    #??
    
    return tps, fps, thresholds
