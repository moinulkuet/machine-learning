# -*- coding: utf-8 -*-
"""
 Multiple Linear regression in python - code example
- Considered mupltiple dependent variables
- Formula:
    
    y = b0 + b1.x1 + b2.x2 + b3.x3 + ..... + bN.xN
    
    y = dependent variable
    x1, x2, ..., xN = Independent variables
    b0 = offset/constant
    b1, b2, ..., bN = coeffiecents
    
    Theory: 5 methods of building a model
        - all-in
        - Backward elimination
        - Forward selection
        - Bi-directional elimination
        - Model comparison
        
Created on Wed Jul 05 14:32:07 2017

@author: Moinul Al-Mamun
"""

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load sample data
data = pd.read_csv("50_Startups.csv")
#independent features 
X = data.iloc[:, :-1].values
# dependent data column
y = data.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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

# fitting multiple linear regression on train dataset
# learn the correlation between the data features
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting the Test set result
y_pred = regressor.predict(X_test)

# Building a optimal model using Backward elimination technique
# Process of selecting set of important variables cosidering influence of the variable
# in prediction

import statsmodels.formula.api as sm
# prepare data for backward elimination
# add a new column (b1x1) to fulfill the equation
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis=1) 

# optimal model -array
X_opt = X[:, [0, 1, 2, 3, 4,5]]
# OLS - ordinary Least Square method
regressor_ols = sm.OLS(endog=y, exog=X_opt).fit()
# look up P value 
regressor_ols.summary()
"""
==============================================================================
                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
const       5.013e+04   6884.820      7.281      0.000      3.62e+04   6.4e+04
x1           198.7888   3371.007      0.059      0.953     -6595.030  6992.607
x2           -41.8870   3256.039     -0.013      0.990     -6604.003  6520.229
x3             0.8060      0.046     17.369      0.000         0.712     0.900
x4            -0.0270      0.052     -0.517      0.608        -0.132     0.078
x5             0.0270      0.017      1.574      0.123        -0.008     0.062
==============================================================================
"""
# remove X2 variable because of high P value
# and repeat  
X_opt = X[:, [0, 1, 3, 4,5]]
# OLS - ordinary Least Square method
regressor_ols = sm.OLS(endog=y, exog=X_opt).fit()
# look up P value 
regressor_ols.summary()

"""
==============================================================================
                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
const       5.011e+04   6647.870      7.537      0.000      3.67e+04  6.35e+04
x1           220.1585   2900.536      0.076      0.940     -5621.821  6062.138
x2             0.8060      0.046     17.606      0.000         0.714     0.898
x3            -0.0270      0.052     -0.523      0.604        -0.131     0.077
x4             0.0270      0.017      1.592      0.118        -0.007     0.061
==============================================================================
"""
# remove X1 variable because of high P value
# and repeat  
X_opt = X[:, [0, 3, 4,5]]
# OLS - ordinary Least Square method
regressor_ols = sm.OLS(endog=y, exog=X_opt).fit()
# look up P value 
regressor_ols.summary()

"""
==============================================================================
                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
const       5.012e+04   6572.353      7.626      0.000      3.69e+04  6.34e+04
x1             0.8057      0.045     17.846      0.000         0.715     0.897
x2            -0.0268      0.051     -0.526      0.602        -0.130     0.076
x3             0.0272      0.016      1.655      0.105        -0.006     0.060
==============================================================================
"""
# remove X2 variable because of high P value
# and repeat  
X_opt = X[:, [0, 3, 5]]
# OLS - ordinary Least Square method
regressor_ols = sm.OLS(endog=y, exog=X_opt).fit()
# look up P value 
regressor_ols.summary()

"""
------------------------------------------------------------------------------
const       4.698e+04   2689.933     17.464      0.000      4.16e+04  5.24e+04
x1             0.7966      0.041     19.266      0.000         0.713     0.880
x2             0.0299      0.016      1.927      0.060        -0.001     0.061
==============================================================================
"""
# remove X2 variable because of high P value
# and repeat  
X_opt = X[:, [0, 3]]
# OLS - ordinary Least Square method
regressor_ols = sm.OLS(endog=y, exog=X_opt).fit()
# look up P value 
regressor_ols.summary()


#visualizing the training result
"""plt.scatter(X_train, y_train, color='red')
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
"""

