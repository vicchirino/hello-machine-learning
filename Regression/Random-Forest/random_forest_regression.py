import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv(r"/Users/victorchirino/Projects/learning-ml/Regression/Random-Forest/Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[: , -1].values
print(X)
print(y)

# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=50, random_state=0)
regressor.fit(X, y)

# Predict a new result
print(regressor.predict([[6.5]]))

# Visualising the Random Forest Regression results (high resolution) 
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color="red")
plt.plot(X_grid, regressor.predict(X_grid), color="blue")
plt.title("Random Forest Regression (RFR)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()