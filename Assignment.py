# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 02:37:31 2021

@author: FARZAN
"""

#Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%
dataset = pd.read_csv("Assignment.csv")
X = dataset.iloc[:,0:3].values
y= dataset.iloc[:,3:4].values
#%%
#Preproccessing for SVR
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(np.reshape(y,(10,1)))
#%%
#Fit Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
print(lin_reg.score(X,y)*100)
#%%
#Visualinsing the Linear Regression results
plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg.predict(X), color = "blue")
plt.title("Linear Regression")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()   
#%%
#Fitting Polynomial Regression to the Dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y )
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
print(lin_reg_2.score(X_poly,y)*100)
#%%
#Visualinsing the Polynomial Regression results
plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = "blue")
plt.title("Polynomial Regression")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()   

#%%
# Fitting SVR to the Dataset
from sklearn.svm import SVR
lin_regt = SVR(kernel = "rbf")
lin_regt.fit(X, y)
print(lin_regt.score(X,y)*100)
#%%
#Fitting RandomForestRegressor To the Dataset
from sklearn.ensemble import RandomForestRegressor
lin_reg = RandomForestRegressor(n_estimators = 1, random_state = 1)
lin_reg.fit(X, y)
print(lin_reg.score(X,y)*100)

#%%
#Visualising the RandomForestRegressor Results
plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg.predict(X), color = "blue")
plt.title("RondomForest Regression)")
plt.xlabel(" Level")
plt.ylabel("Salary")
plt.show()
