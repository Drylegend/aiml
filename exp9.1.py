import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
X_full = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]

X=X_full[:,5].reshape(-1,1)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def lowess(X, y, f=0.5, kernel='gaussian', n_neighbors=5):
    
    y_pred = np.zeros_like(y)

    
    for i in range(len(y)):
        
        distances = np.linalg.norm(X - X[i], axis=1) 
        threshold = np.quantile(distances, f)
        idx = distances <= threshold 
        subset = X[idx]
        subset_y = y[idx]

    
            
        if kernel == 'gaussian':
            weights = np.exp(-(distances[idx] / threshold) ** 2)
        elif kernel == 'triangular':
            weights = np.maximum(1 - distances[idx] / threshold, 0)

    
        weights /= np.sum(weights)
        reg = LinearRegression()
        reg.fit(subset, subset_y, sample_weight=weights)

    
        y_pred[i] = reg.predict(X[i].reshape(1, -1))[0]

    
    mse = mean_squared_error(y, y_pred)
    return y_pred, mse



y_pred, mse = lowess(X, y, f=0.3, kernel='gaussian')



sort_idx = np.argsort(X[:, 0])
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label="Actual", s=10, alpha=0.6)
plt.plot(X[sort_idx], y_pred[sort_idx], color='red', label="LOWESS Prediction", linewidth=2)
plt.xlabel("Average Number of Rooms (RM)")
plt.ylabel("Median Home Value (MEDV)")
plt.title("LOWESS Smoothing on Boston Housing")
plt.legend()
plt.grid(True)
plt.show()

print("Mean squared error:", mse)
