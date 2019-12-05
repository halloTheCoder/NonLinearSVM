import numpy as np

class MinMaxScaler:
    """Transform features by scaling each feature to a given range.
    This estimator scales and translates each feature individually such
    that it is in the given range on the training set, e.g. between
    zero and one.
    The transformation is given by::
        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min
    where min, max = feature_range.
    The transformation is calculated as::
        X_scaled = scale * X + min - X.min(axis=0) * scale
        where scale = (max - min) / (X.max(axis=0) - X.min(axis=0))
    This transformation is often used as an alternative to zero mean,
    unit variance scaling.
    
    Parameters
    ----------
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.
    """
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
    
    def fit(self, X):
        """Compute the minimum and maximum to be used for later scaling.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.
        """
        data_min = np.nanmin(X, axis=0)
        data_max = np.nanmax(X, axis=0)
        
        data_range = (data_max - data_min)
        
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / data_range
        self.min_ = self.feature_range[0] - data_min * self.scale_
        self.data_range_ = data_range
        self.data_min_ = data_min
        self.data_max_ = data_max
    
    def transform(self, X):
        """Scale features of X according to feature_range.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed.
        
        Returns
        -------
        Xt : array-like of shape (n_samples, n_features)
            Transformed data.
        """
        X = np.asarray(X, dtype=np.float64)
        X *= self.scale_
        X += self.min_
        return X
    
    def fit_transform(self, X):
        """Compute the minimum and maximum and Scale features of X according to feature_range.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed.
        
        Returns
        -------
        Xt : array-like of shape (n_samples, n_features)
            Transformed data.
        """
        X = np.asarray(X, dtype=np.float64)
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        """Undo the scaling of X according to feature_range.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed.
            
        Returns
        -------
        Xt : array-like of shape (n_samples, n_features)
            Transformed data.
        """
        X = np.asarray(X, dtype=np.float64)
        X -= self.min_
        X /= self.scale_
        return X


class RobustScaler:
    """Scale features using statistics that are robust to outliers.
    This Scaler removes the median and scales the data according to
    the quantile range (defaults to IQR: Interquartile Range).
    The IQR is the range between the 1st quartile (25th quantile)
    and the 3rd quartile (75th quantile).
    Centering and scaling happen independently on each feature by
    computing the relevant statistics on the samples in the training
    set. Median and interquartile range are then stored to be used on
    later data using the ``transform`` method.
    
    Parameters
    ----------
    with_centering : boolean, True by default
        If True, center the data before scaling.
    with_scaling : boolean, True by default
        If True, scale the data to interquartile range.
    quantile_range : tuple (q_min, q_max), 0.0 < q_min < q_max < 100.0
        Default: (25.0, 75.0) = (1st quantile, 3rd quantile) = IQR
        Quantile range used to calculate ``scale_``.
    """
    def __init__(self, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0)):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
    
    def fit(self, X):
        """Compute the median and quantiles to be used for scaling.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to compute the median and quantiles
            used for later scaling along the features axis.
        """
        q_min, q_max = self.quantile_range
        self.center_ = np.nanmedian(X, axis=0) if self.with_centering else None
        
        if self.with_scaling:
            quantiles = []
            for feature_idx in range(X.shape[1]):
                column_data = X[:, feature_idx]
                quantiles.append(np.nanpercentile(column_data, self.quantile_range))

            quantiles = np.transpose(quantiles)
            self.scale_ = quantiles[1] - quantiles[0]
        else:
            self.scale_ = None
        
        return self
    
    def transform(self, X):
        """Center and scale the data.
        Parameters
        ----------
        X : array-like
            The data used to scale along the specified axis.
        """
        X = np.asarray(X, dtype=np.float64)
        
        if self.with_centering:
            X -= self.center_
        if self.with_scaling:
            X /= self.scale_
            
        return X
    
    def fit_transform(self, X):
        """Compute the median and quantiles to be used for scaling and center and scale the data.
        
        Parameters
        ----------
        X : array-like
            The data used to scale along the specified axis.
        """
        X = np.asarray(X, dtype=np.float64)
        
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        """Scale back the data to the original representation
        
        Parameters
        ----------
        X : array-like
            The data used to scale along the specified axis.
        """
        X = np.asarray(X, dtype=np.float64)
        
        if self.with_scaling:
            X *= self.scale_
        if self.with_centering:
            X += self.center_
        
        return X
