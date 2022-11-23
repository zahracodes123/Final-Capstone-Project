#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 22:23:50 2022

@author: zahra
"""

# For the final project I have decided to use Yfin as a source instead of using the s&p 500 data from Kaggle. Firstly, because  
# the original dataset being used only gave the prices of the stocks as individual stocks and
# the aim is to predict the performance of the benchmark as a whole intead of predicting the 
# prices of stocks only. Secondly Yfinance is more user friendy and shows the data in the form
# of charts and visuals as well and it is easier to visualise their data and see how it compares to
# the results in this project 
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn import neighbors
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

data=yf.download("^GSPC", start="2019-11-01", end="2022-11-01")
data_financials=pd.DataFrame(data)
data_financials.to_csv("S&P500.csv")

d=pd.read_csv('S&P500.csv')
df=d.dropna()
df.set_index("Date",inplace=True)
column=df.pop("Volume")
df.insert(0,"Volume",column)

df=pd.get_dummies(df)
x=df.iloc[:,0:5].values
y=df.iloc[:,5]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
 
#In order to use the data for analysis , it is important to clean the data first. So the data has
#had all of its Na values removed and the date has beeen set as the index in order to locate data 
#and use it more efficiently. The stock price from the past three years have been used as they 
#cover the period from before COVID-19 to now. The model can also be adjusted and prices from a longer 
#time period can be used by exporting data from more years. The data has also had the pd.get_dummies
#feature applied to it so that in case there is any categorical variable it is converted to numeric data.
#The features that will be used to predict the independant variable which is 'Adjusted Close' are 'High', 'Low',
#'Open', 'Close' and 'Volume'. These variables are basically showing stock market prices on a daily basis and the
#trading activity which is volume. The adjusted close price is the price of the stock after all the corporate actions have 
#been performed and this is our y variable and will be used as the variable being predicted. Then the data is split into 
#a train test ratio which is 70-30.


model=RandomForestRegressor(n_estimators=10, random_state=0)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(y_pred)
print(y_pred.shape)
RMSE1=float(format(np.sqrt(mean_squared_error(y_test, y_pred)),'.3f'))
RFR2=r2_score(y_test, y_pred)

#The random forest regression model is used. In this case the hyper parameters have not been 
#adjusted as this is a skeletal code and will be improved in the final analysis. Hyperparameters 
#are adjusted in order to improve the accuracy and fitting of the model. 


regressor=LinearRegression()
regressor.fit(x_train,y_train)
predict_y=regressor.predict(x_test)
RMSE2=mean_squared_error(y_test, predict_y)
LRR2=r2_score(y_test,predict_y)

#The second regression model being used is Multiple Linear Regression. Here the dependant 
#variable is 'Adjusted Close' and the independant variables are 'High', 'Low', 'Open', 'Close',
#'volume'. Since we are using multiple variables to predict one independant variable , multiple linear 
#regression is being used. 


scaler=MinMaxScaler(feature_range=(0,1))
x_train_scaled=scaler.fit_transform(x_train)
x_train=pd.DataFrame(x_train_scaled)
x_test_scaled=scaler.fit_transform(x_test)
x_test=pd.DataFrame(x_test_scaled)


model=neighbors.KNeighborsRegressor(n_neighbors=5)
model.fit(x_train,y_train)
predd=model.predict(x_test)
RMSE3=sqrt(mean_squared_error(y_test,predd))
KNR2=r2_score(y_test, predd)
#The third algorithm that will be used is the KN Regressor. The number of neighbours being 
#used is 5 which is also the default sett The data is first scaled using the min max scaler which reduces 
#the values of the data to a number between 0 and 1 without changing the distribution of the data. Then the model is 
#fit on the data to predict the stock prices.

print(RMSE1)
print(RFR2)
print(RMSE2)
print(LRR2)
print(RMSE3)
print(KNR2)

#In order to test which regression algorithm is more accurate , two measures are used. One measure is the RMSE and the 
#other is the R^2 values. When we run the results, we see that KN regression has the highest RMSE and multiple linear regression has the lowest.
#Higher RMSE values indicate that the model is not very good at predicting the values it is supposed to predict. In this case , because our model 
#is a prediction model, higher RMSE is not ideal. R^2 value of Multiple Linear Regression is one , that inicates that the model fits the data 
#perfectly. Second best score is Random Forest Regression's. The worst scores for both measures are for the KN regression model. Looking at this 
#initial result, it can be concluded that multiple linear regression is the best model for stock price prediction and KN Neighbors is the least 
#accurate. 