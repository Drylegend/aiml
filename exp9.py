import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def lowess(X, y, f=0.5, kernel='gaussian', n_neighbors=5):
    """
    Apply the LOWESS algorithm to the given input-output data.
    Parameters:
    X (array-like): Input data with shape (n_samples, n_features).
    y (array-like): Output data with shape (n_samples,).
    f (float): Fraction of the data used for fitting.
    kernel (str): Kernel function used to compute the weights. Can be either 'gaussian' or 'triangular'.
    n_neighbors (int): Number of neighbors used for smoothing.
    Returns:
    y_pred (array-like): Predicted output data with shape (n_samples,).
    """

    
    y_pred = np.zeros_like(y)

    
    for i in range(len(y)):
        
        distances = np.linalg.norm(X - X[i], axis=1)  
        idx = distances < (distances.max() - distances.min()) * f  
        subset = X[idx]
        subset_y = y[idx]

    
    reg = LinearRegression()
    reg.fit(subset, subset_y)

    
    if kernel == 'gaussian':
        weights = np.exp(-(np.abs(X - X[i]) / (X.max() - X.min()) * f)**2)
    elif kernel == 'triangular':
        weights = np.maximum(1 - np.abs(X - X[i]) / (X.max() - X.min()) * f, 0)

    
    weights /= np.sum(weights)

    
    y_pred[i] = np.sum(weights * reg.predict(X[i][None, :]))

    
    mse = mean_squared_error(y, y_pred)
    return y_pred, mse



y_pred, mse = lowess(X, y)



plt.scatter(X[:, 5], y, s=10)
plt.xlabel('RM (median value of owner-occupied homes in $1000s)')
plt.ylabel('MEDV (median value of owner-occupied homes in $1000s)')


plt.plot(X[:, 5], y_pred, color='red', linewidth=2)
plt.show()

print("Mean squared error:", mse)
