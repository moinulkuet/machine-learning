# -*- coding: utf-8 -*-
"""
Simple Linear regression in python - code example
- Considered single dependent variable
- Formula of simple linear regression model
    y = b0 + b1.x
    
    y = dependent variable
    x = Independent variable
    b0 = offset/constant
    b1 = coeffiecent
    
Created on Wed Jul 05 14:32:07 2017

@author: Moinul Al-Mamun
"""

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load sample data
data = pd.read_csv("Salary_Data.csv")
#independent features 
X = data.iloc[:, :-1].values
# dependent data column
y = data.iloc[:, 1].values

#split train and test dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# fitting linear regression on train dataset
# learn the correlation between the data features
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting the Test set result
y_pred = regressor.predict(X_test)

#visualizing the training result
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train))
plt.title("Salary vs years of experience (Training set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()

#visualizing the test set result
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train))
plt.title("Salary vs years of experience (Test set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()

