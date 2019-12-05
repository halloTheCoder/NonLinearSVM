import numpy as np
import cvxopt

from kernels import linear_kernel, polynomial_kernel, gaussian_kernel

class BaseSVC:
    """The Base class for Support Vector Machine classifier,
    which is used by BinarySVC and SVC(multiclass).
    
    Parameters
    -----------
    C : float, optional (default=1.0)
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive. The penalty
        is a squared l2 penalty.
    kernel : string, optional (default='rbf')
        Specifies the kernel type to be used in the algorithm.
        It must be one of 'linear', 'poly', 'rbf'.
    degree : int, optional (default=3)
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.
    gamma : {'scale', 'auto'} or float, optional (default='scale')
        Kernel coefficient for 'rbf' and 'poly'.
        - if ``gamma='scale'`` (default) is passed then it uses
          1 / (n_features * X.var()) as value of gamma,
        - if 'auto', uses 1 / n_features.
    coef0 : float, optional (default=0.0)
        Independent term in kernel function.
        It is only significant in 'poly'.
    class_weight : {array-like, 'balanced'}, optional
        Set the parameter C of class i to class_weight[i]*C for
        SVC. If not given, all classes are supposed to have weight one.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``
    """
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0):
        if gamma == 0:
            msg = ("The gamma value of 0.0 is invalid. Use 'auto' to set gamma to a value of 1 / n_features.")
            raise ValueError(msg)
            
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.C = C
        
        self.kernels = {'rbf' : gaussian_kernel, 'linear' : linear_kernel, 'poly' : polynomial_kernel}
        self.kernel_function = self.kernels[self.kernel]


class BinarySVC(BaseSVC):
    """The Base class for Support Vector Machine classifier,
    which is used by BinarySVC and SVC(multiclass).
    
    Parameters
    -----------
    C : float, optional (default=1.0)
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive. The penalty
        is a squared l2 penalty.
    kernel : string, optional (default='rbf')
        Specifies the kernel type to be used in the algorithm.
        It must be one of 'linear', 'poly', 'rbf'.
    degree : int, optional (default=3)
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.
    gamma : {'scale', 'auto'} or float, optional (default='scale')
        Kernel coefficient for 'rbf' and 'poly'.
        - if ``gamma='scale'`` (default) is passed then it uses
          1 / (n_features * X.var()) as value of gamma,
        - if 'auto', uses 1 / n_features.
    coef0 : float, optional (default=0.0)
        Independent term in kernel function.
        It is only significant in 'poly'.
    """
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0):
        super().__init__(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0
        )
        
    def fit(self, X, y):
        """Fit the SVM model for binary classification
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target values, binary labels {-1, +1}
        """
        if isinstance(self.gamma, str):
            if self.gamma == 'scale':
                self._gamma = 1. / (X.shape[1] * X.var()) if X.var() != 0 else 1.0
                
            elif self.gamma == 'auto':
                self._gamma = 1. / X.shape[1]
            
            else:
                msg = ("The gamma can only be 'auto' or 'scale'.")
                raise ValueError(msg)
        else:
            self._gamma = self.gamma
        
        self.kwargs = {'degree' : self.degree, 'gamma' : self._gamma, 'coef0' : self.coef0}
        
        X, y = np.asarray(X), np.asarray(y, dtype=np.int64)
        
        n_samples, n_features = X.shape

        # Gram Matrix
        K = np.zeros(shape = (n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i][j] = self.kernel_function(X[i], X[j], **self.kwargs)
        
        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples), 'd')
        b = cvxopt.matrix(0.0)
        
        tmp1 = np.diag(np.ones(n_samples) * -1)
        tmp2 = np.identity(n_samples)
        G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
        tmp1 = np.zeros(n_samples)
        tmp2 = np.ones(n_samples) * self.C
        h = cvxopt.matrix(np.hstack((tmp1, tmp2)))
        
        cvxopt.solvers.options['show_progress'] = False
        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
#         print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == 'linear':
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None
        
        return self
    
    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel_function(X[i], sv, **self.kwargs)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X, threshold=0.0):
        return np.sign(self.project(X) + threshold)


class SVC(BinarySVC):
    """The Support Vector Machine classifier.
    Uses cvxopt to solve the quadratic optimization problem.
    
    Parameters
    -----------
    C : float, optional (default=1.0)
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive. The penalty
        is a squared l2 penalty.
    kernel : string, optional (default='rbf')
        Specifies the kernel type to be used in the algorithm.
        It must be one of 'linear', 'poly', 'rbf'.
    degree : int, optional (default=3)
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.
    gamma : {'scale', 'auto'} or float, optional (default='scale')
        Kernel coefficient for 'rbf' and 'poly'.
        - if ``gamma='scale'`` (default) is passed then it uses
          1 / (n_features * X.var()) as value of gamma,
        - if 'auto', uses 1 / n_features.
    coef0 : float, optional (default=0.0)
        Independent term in kernel function.
        It is only significant in 'poly'.
    class_weight : {dictionary, 'balanced'}, optional
        Set the parameter C of class i to class_weight[i]*C for
        SVC. If not given, all classes are supposed to have weight one.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``
    """
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, class_weight=None):
        super().__init__(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0
        )
        self.class_weight = class_weight
        
    def fit(self, X, y):
        """Fit the SVM model according to the given training data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
            NOTE: All the columns should be normalized and label-encoded i.e.
            no column should be of dtype ``str``
        y : array-like, shape (n_samples,)
            Target values (class labels in classification)
        """
        
        X, y = np.asarray(X), np.asarray(y, dtype=np.int64)
        n_samples = X.shape[0]
        labels = np.unique(y)
        n_labels = len(labels)
        
        self.ind2lab = {i:unique_label for i, unique_label in enumerate(labels)}
        
        if self.class_weight is None:
            self.class_weight = dict(zip(labels, np.ones_like(labels)))
            
        if isinstance(self.class_weight, str):
            if self.class_weight == 'balanced':
                self.class_weight_ = dict(zip(labels, (n_samples/(n_labels * np.bincount(y)))))
            else:
                msg = ("The class weight parameter can only be 'balanced'.")
                raise ValueError(msg)
        else:
            self.class_weight_ = self.class_weight
        
        if n_labels == 1:
            msg = ("No of labels in classification task should be more than 1.")
            raise ValueError(msg)
        
        else:
            self.classifiers = []
            for unique_label in labels:
                y_enc_bin = self.encode(y, unique_label)
                clf = BinarySVC(
                    C=(self.C * self.class_weight_[unique_label]),
                    kernel=self.kernel,
                    degree=self.degree,
                    gamma=self.gamma,
                    coef0=self.coef0
                )
                self.classifiers.append(clf.fit(X, y_enc_bin))
        
        return self.classifiers
    
    def encode(self, y, unique_label):
        """Convert multiclass labels to binary label in the
        form of {-1, +1}, given an unique label.
        
        Parameters
        ----------
        y : array_like, shape (n_samples, 1)
            Multiclass labels to be encoded to binary labels
        """
        y_enc = np.ones_like(y)
        y_enc[np.logical_not(y==unique_label)] = -1
        return y_enc
        
    def predict(self, X, classifiers=None, threshold=0.0):
        """Predict the unseen data using OVA method for multiclass classification
        using OVA strategy using n_labels classifiers
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        classifiers : list
                      n_labels classifiers to predict the multi-class
        threshold : int, default=0.0
        Returns
        ----------
        y : array-like, shape (n_sample, 1)
            Predicted values, multiclass labels
        """
        if not classifiers:
            classifiers = self.classifiers

        return np.asarray([self.ind2lab[np.argmax([clf.project(x.reshape(1, -1)) for clf in classifiers])] for x in X], dtype=np.int64)


def gen_non_lin_separable_data():
    mean1 = [-1, 2]
    mean2 = [1, -1]
    mean3 = [4, -4]
    mean4 = [-4, 4]
    cov = [[1.0,0.8], [0.8, 1.0]]
    X1 = np.random.multivariate_normal(mean1, cov, 50)
    X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, 50)
    X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
    y2 = np.ones(len(X2)) * -1
    return X1, y1, X2, y2

def split_train(X1, y1, X2, y2):
    X1_train = X1[:90]
    y1_train = y1[:90]
    X2_train = X2[:90]
    y2_train = y2[:90]
    X_train = np.vstack((X1_train, X2_train))
    y_train = np.hstack((y1_train, y2_train))
    return X_train, y_train

def split_test(X1, y1, X2, y2):
    X1_test = X1[90:]
    y1_test = y1[90:]
    X2_test = X2[90:]
    y2_test = y2[90:]
    X_test = np.vstack((X1_test, X2_test))
    y_test = np.hstack((y1_test, y2_test))
    return X_test, y_test

import matplotlib.pyplot as pl

def test_non_linear():
    X1, y1, X2, y2 = gen_non_lin_separable_data()
    X_train, y_train = split_train(X1, y1, X2, y2)
    X_test, y_test = split_test(X1, y1, X2, y2)

    clf = BinarySVC(kernel='rbf')
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    correct = np.sum(y_predict == y_test)
    print("%d out of %d predictions correct" % (correct, len(y_predict)))

    plot_contour(X_train[y_train==1], X_train[y_train==-1], clf)

def plot_margin(X1_train, X2_train, clf):
    def f(x, w, b, c=0):
        # given x, return y such that [x,y] in on the line
        # w.x + b = c
        return (-w[0] * x - b + c) / w[1]

    pl.plot(X1_train[:,0], X1_train[:,1], "ro")
    pl.plot(X2_train[:,0], X2_train[:,1], "bo")
    pl.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c="g")

    # w.x + b = 0
    a0 = -4; a1 = f(a0, clf.w, clf.b)
    b0 = 4; b1 = f(b0, clf.w, clf.b)
    pl.plot([a0,b0], [a1,b1], "k")

    # w.x + b = 1
    a0 = -4; a1 = f(a0, clf.w, clf.b, 1)
    b0 = 4; b1 = f(b0, clf.w, clf.b, 1)
    pl.plot([a0,b0], [a1,b1], "k--")

    # w.x + b = -1
    a0 = -4; a1 = f(a0, clf.w, clf.b, -1)
    b0 = 4; b1 = f(b0, clf.w, clf.b, -1)
    pl.plot([a0,b0], [a1,b1], "k--")

    pl.axis("tight")
    pl.show()

def plot_contour(X1_train, X2_train, clf):
    pl.plot(X1_train[:,0], X1_train[:,1], "ro")
    pl.plot(X2_train[:,0], X2_train[:,1], "bo")
    pl.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c="g")

    X1, X2 = np.meshgrid(np.linspace(-6,6,50), np.linspace(-6,6,50))
    X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
    Z = clf.project(X).reshape(X1.shape)
    pl.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
    pl.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
    pl.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')

    pl.axis("tight")
    pl.show()

def test():
    test_non_linear()
