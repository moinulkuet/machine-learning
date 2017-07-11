# -*- coding: utf-8 -*-
"""
 Random Forest Regression in python - code example
- similar of Decision tree method
- A version of Ensemble learning - other Ensemble learning are gradient Boosting
- Ensemble - means taking multiple algorithms or same algorithms for multiple time and
 put them together for making some thing more powerful than the original one
- Steps:
    - Step1: Pick at random k data points from the training set
    - Step2: Build the Decision tree using the k data points
    - Step3: Choose the number Ntree of trees you want to build and repeat steps 1 & 2 
    - Step4: For a new data point, predict the value of Y for NTree of tree and 
    assign Y value for all of the tree average
        - Not predicting for one tree, it is predicting for forest of tree. 
        Taking the average of many predictions, so it improves the accuracy
 
- Decision tree called "CART", stands for Classification And Regression Tree
- A non-continuous regression. | Linear, Polynomial and SVR was continuous regression
- In Decision tree - Average the independent variable for the interval.
- Decision Tree is not interesting model for 1D, but very interesting for multiple Dimensions


- Regression model:
        - Linear regression model   [Linear and Multiple linear regression]
        - Non-linear regression model [Polynomial linear regression, SVR]
        - Non-linear Non-continuous regression model [Decision tree]
        - Non-linear non-continuous ensembble regression model [Random Forest]

Created on Wed Jul 06 14:32:07 2017
@author: Moinul Al-Mamun
"""
# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
# n_estimators - Number of trees - important parameters
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

# 100 trees
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X, y)

# 300 trees
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X, y)


# Predicting a new result
# 10 trees - 167k
# 100 trees - 158k
# 300 trees - 160k -- wow!!
y_pred = regressor.predict(6.5)

# Visualising the Random Forest Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()