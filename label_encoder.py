import numpy as np
import pandas as pd

class LabelEncode:
    """
    Encode target labels with value between 0 and n_classes-1.
    """
    def __init__(self):
        self.transform_dict = dict()
        self.cnt = 0
        
    def fit(self, X):
        if not isinstance(X, (list, np.ndarray, pd.Series)):
            raise TypeError("X assumed to be a numpy array, pandas Series or a list"
                            " ,but got {0}".format(type(X)))
        for item in X:
            if item not in self.transform_dict:
                self.transform_dict[item] = self.cnt
                self.cnt += 1
        
    def transform(self, X):
        if not isinstance(X, (list, np.ndarray, pd.Series)):
            raise TypeError("X assumed to be a numpy array, pandas Series or a list"
                            " ,but got {0}".format(type(X)))
        return [self.transform_dict[item] for item in X]
        
    def fit_transform(self, X):
        if not isinstance(X, (list, np.ndarray, pd.Series)):
            raise TypeError("X assumed to be a numpy array, pandas Series or a list"
                            " ,but got {0}".format(type(X)))
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        if not isinstance(X, (list, np.ndarray, pd.Series)):
            raise TypeError("X assumed to be a numpy array, pandas Series or a list"
                            " ,but got {0}".format(type(X)))
        reverse_transform_dict = {v:k for k, v in self.transform_dict.items()}
        return [reverse_transform_dict[i] for i in X]
    
    def classes_(self):
        return list(self.transform_dict.keys())
    
#     def save(self, filepath=None):
#         if not filepath:
#             filepath = 'LabelEncoder.txt'
        
#         with open(filepath, 'w') as f:
#             f.write('{\n')
#             for item, i in self.transform_dict.items():
#                 f.write('\t' + str(item) + ':' + str(i) + ',\n')
#             f.write('}\n')
    
#     @staticmethod
#     def load(filepath=None):
#         if not filepath:
#             raise FileNotFoundError
        
#         transform_dict = dict()
        
#         with open(filepath, 'r') as f:
#             for line in f.readlines()[1:-1]:
#                 print(line.strip().split(':'))
#                 transform_dict.update(*line.strip().split(':'))
            
#         return LabelEncode(transform_dict)
