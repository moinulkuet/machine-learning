# -*- coding: utf-8 -*-
"""
 Polynomial Linear regression in python - code example
- Considered one dependent variables with polynomial feature
- Formula:
    
    y = b0 + b1.x1^2 + b2.x1^2 + b3.x1^2 + ..... + bN.x1^n
    - parabolic effect, curvature, circle, etc 
    
    y = dependent variable
    x1 = Independent variables
    b0 = offset/constant
    b1, b2, ..., bN = coeffiecents

Why it called Linear? 
- not because of X value, but it is bacause of b0, b1, b2,... coeffecient's linear nature

Created on Wed Jul 05 14:32:07 2017
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

# Encoding categorical data
# Encoding the Independent Variable
"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# avoiding dummy variable trap
# it will remove the first column: Actually we donot need to do that
# because linear regression model take care of this. but need to keep in mind this case 
X = X[:, 1:]

#split train and test dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
"""

# fitting linear regression on dataset, this time we are not spliliting Dataset, because 
# our dataset is very small.
from sklearn.linear_model import LinearRegression
lin_regressor = LinearRegression()
lin_regressor.fit(X, y)

"""# fitting polynimal regression
# polynomial comes as preprocessing features
from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree=2)   # x^0, x^1, x^2 
X_poly = poly_regressor.fit_transform(X)
#it includes an extra constant column for liner regression.
# now apply linear regression on polynomial data
lin_regressor_2 = LinearRegression()
lin_regressor_2.fit(X_poly, y)

# experiment 3 degree
from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree=3)   # x^0, x^1, x^2, x^3 
X_poly = poly_regressor.fit_transform(X)
#it includes an extra constant column for liner regression.
# now apply linear regression on polynomial data
lin_regressor_2 = LinearRegression()
lin_regressor_2.fit(X_poly, y)"""

# experiment 4 degree
from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree=4)   # x^0, x^1, x^2, x^3, x^4  
X_poly = poly_regressor.fit_transform(X)
#it includes an extra constant column for liner regression.
# now apply linear regression on polynomial data
lin_regressor_2 = LinearRegression()
lin_regressor_2.fit(X_poly, y)

#visualizing the linear regression result
plt.scatter(X, y, color='red')
plt.plot(X, lin_regressor.predict(X), color="blue")
plt.title("Truth or bluf - Salary (Linear regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()
# if we carefully examine the graph, it is understandable that it was not a good fit, not good prediction

#visualizing the polynomial regression result
#X_grid = np.arange(min(X), max(X), 0.1)
plt.scatter(X, y, color='red')
plt.plot(X, lin_regressor_2.predict(poly_regressor.fit_transform(X)), color="blue")
plt.title("Truth or bluf - Salary (Polynomial regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

"""
Polynomial give much better result than the linear regression
Higher degree of polynomial fit better with data, accurate prediction
"""

# prediction
lin_regressor.predict(6.5) # array([ 330378.78787879])
lin_regressor_2.predict(poly_regressor.fit_transform(6.5)) #array([ 158862.45265153])
