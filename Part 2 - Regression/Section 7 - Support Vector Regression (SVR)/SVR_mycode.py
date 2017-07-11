# -*- coding: utf-8 -*-
"""
 Non Linear regression - Support vector Regression (SVR) in python - code example
- Formula:
     
Created on Wed Jul 06 14:32:07 2017
@author: Moinul Al-Mamun
"""

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load sample data
data = pd.read_csv("Position_Salaries.csv")
#independent features 
X = data.iloc[:, 1:2].values # make a matrix, always better (10, 1)
# dependent data column
y = data.iloc[:, 2].values # vector (10)

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
# in Linear regression feature scaling was in built, but in SVR it have to do explicitly
# for getting good result.
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

                            
# fitting SVR to the dataset
from sklearn.svm import SVR
# important params - kernel: options- linear, poly, rbf, sigmoid, etc
# rbf is default and non-linear, so we choose it
regressor = SVR(kernel="rbf") 
regressor.fit(X, y)

# prediction
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
# without scaling - array([ 130001.55760156]) - wow... very low than polynomial
# after array([ 170370.0204065]) ... great result!

#visualizing the linear regression result
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color="blue")
plt.title("Truth or bluf - Salary (SVR regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()
# if we carefully examine the graph, it is understandable that it was not a good fit, not good prediction

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
