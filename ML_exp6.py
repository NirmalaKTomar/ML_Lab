# -*- coding: utf-8 -*-
"""ML exp-6 """

print("Exp-6,Write a prg to implement Linear Regression using any app.dataset")

print("implementing Linear Regression on boston house data price\n")

#importing all necessary libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

"""Load and return **boston house price dataset** from  
(https://scikit-learn.org/stable/datasets/index.html#boston-dataset)
"""

from sklearn.datasets import load_boston
X, y = load_boston(return_X_y=True)

X.shape

y[:5]

#Splitting the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25) 
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

# creating an object of LinearRegression
regr = LinearRegression()

#training data
regr.fit(X_train, y_train) 
y_pred = regr.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test,y_pred)
print(rmse)
print(r2)

fig,ax = plt.subplots()
ax.scatter(y_test,y_pred)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
polynomial_features= PolynomialFeatures(degree=2)
x_poly = polynomial_features.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(x_poly, y, test_size = 0.25) 
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
model = LinearRegression()
model.fit(X_train, y_train)
y_poly_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test,y_poly_pred))
r2 = r2_score(y_test,y_poly_pred)
print(rmse)
print(r2)

fig,ax = plt.subplots()
ax.scatter(y_test,y_poly_pred)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()