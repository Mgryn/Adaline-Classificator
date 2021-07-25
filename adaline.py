# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class AdalineGD(object):
    """ADAptive LInear NEuron classifier

    Parameters:
        eta : float, learning rate (0 ; 1)
        n_iter : int, passes over the training dataset
        random_state : random number generator seed 
        w_: ndarray, weights after fitting
        cost_ : sum-of-squares cost function value in each epoch
    """
    
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        """Init.

        Parameters:
            eta : float, learning rate (0 ;1)
            n_iter : int, passes over the training dataset
            random_state : random number generator seed
        """
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        """Fits model on the training data.

        Parameters:
        X : ndarray, training vector
        y : ndarray, target values

        Returns:
        self : class, fitted classifier
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size = 1 + X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self
    
    def net_input(self, X):
        """Calculates net input."""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        """Computes linear activation."""
        return X
    
    def predict(self, X):
        """Returns class label after unit step."""
        return np.where(self.activation(self.net_input(X) >= 0.0 ,1 , -1))
    

# import Iris dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                 'iris/iris.data', header = None)

# select setosa and versicolor species
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
# select sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

# plot cost against number of epochs for two different learning rates:
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_)+1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')
ada2 = AdalineGD(n_iter = 10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')
plt.show()
