import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv(r"/Users/victorchirino/Projects/hello-machine-learning/Regression/Decision-Tree/Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[: , -1].values
print(X)
print(y)

# Training the Decision Tree Regression model on the whole dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# Predict a new result
regressor.predict([[6.5]])

# Visualising the Decision Tree Regression results (high resolution) 
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color="red")
plt.plot(X_grid, regressor.predict(X_grid), color="blue")
plt.title("Decision Tree Regression (DTR)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
