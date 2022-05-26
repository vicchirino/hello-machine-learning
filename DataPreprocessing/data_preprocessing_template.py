# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'/Users/victorchirino/Projects/learning-ml/DataPreprocessing/Data.csv')

X = dataset.iloc[:, :-1].values # : means range. I.e: 2:4. 
y = dataset.iloc[:, -1].values # Dependent variable vector

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)