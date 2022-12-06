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
from patsy import dmatrices 
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae

data=yf.download("^GSPC", start="2019-11-01", end="2022-11-01")
data_financials=pd.DataFrame(data)
data_financials.to_csv("S&P500.csv")

d=pd.read_csv('S&P500.csv')
df=d.dropna()
df.set_index("Date",inplace=True)
column=df.pop("Volume")
df.insert(0,"Volume",column)
df.rename(columns={'Adj Close':'AdjClose'}, inplace=True)

y,X=dmatrices('AdjClose~Volume+Open+High+Low+Close',data=df,return_type='dataframe')
vif=pd.DataFrame()
vif['variable']=X.columns
vif['VIF']=[variance_inflation_factor(X.values,i)for i in range(X.shape[1])]
print(vif)

#The VIF Score is used to test the collinearity of the independant variables to test which variables are 
#statistically significant. In this case all of them have high collinearity measures except Volume

df=pd.get_dummies(df)
x=df.iloc[:,0:5]
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
RFMAE=mae(y_test,y_pred)



regressor=LinearRegression()
regressor.fit(x_train,y_train)
predict_y=regressor.predict(x_test)
RMSE2=mean_squared_error(y_test, predict_y)
LRR2=r2_score(y_test,predict_y)
LRMAE=mae(y_test,predict_y)


#The second regression model being used is Multiple Linear Regression. Here the dependant 
#variable is 'Adjusted Close' and the independant variables are 'High', 'Low', 'Open', 'Close',
#'volume'. Since we are using multiple variables to predict one independant variable , multiple linear 
#regression is being used. 

scaler=MinMaxScaler(feature_range=(0,1))
x_train_scaled=scaler.fit_transform(x_train)
x1_train=pd.DataFrame(x_train_scaled)
x_test_scaled=scaler.fit_transform(x_test)
x1_test=pd.DataFrame(x_test_scaled)


model=neighbors.KNeighborsRegressor(n_neighbors=15)
model.fit(x1_train,y_train)
predd=model.predict(x1_test)
RMSE3=sqrt(mean_squared_error(y_test,predd))
KNR2=r2_score(y_test, predd)
KNMAE=mae(y_test,predd)


#The third algorithm that will be used is the KN Regressor. The number of neighbours being 
#used is 5 which is also the default sett The data is first scaled using the min max scaler which reduces 
#the values of the data to a number between 0 and 1 without changing the distribution of the data. Then the model is 
#fit on the data to predict the stock prices.The reason why we use min max scalewr to scale the data for k nearest neighbours is that k nearest 
#neighbours is dependant upon the distance between the points. Really extreme values can thus have asignificant impact on results 


print("Random Forest RMSE is = " + str(RMSE1))
print("Random Forest R Square is =" + str(RFR2))
print("Random Forest Mean Absolute Error is="+str(RFMAE))
print("Linear Regression RMSE is ="+str(RMSE2))
print("Linear Regression R Square is="+str(LRR2))
print("Linear Regression MAE is="+str(LRMAE))
print("knn RMSE is="+str(RMSE3))
print("Knn R Square is="+str(KNR2))
print("Knn MAE is="+str(KNMAE))



#In order to test which regression algorithm is more accurate , three measures are used. One measure is the RMSE, the second is MAE and the 
#third is the R^2. When we run the results, we see that KN regression has the highest RMSE and Multiple Linear Regression has the lowest.
#Higher RMSE values indicate that the model is not very good at predicting the values it is supposed to predict. In this case , because our model 
#is a prediction model, higher RMSE is not ideal. R^2 value of Multiple Linear Regression is one , that inicates that the model fits the data 
#perfectly. Second best score is Random Forest Regression's. The worst scores for both measures are for the KN Regression Model.KNN also has the 
#highest MAE and Linear Regression has the lowest. Looking at this 
#initial result, it can be concluded that multiple linear regression is the best model for stock price prediction and KN Neighbors is the least 
#accurate. 

fig, (ax1,ax2,ax3)=plt.subplots(3)

ax1.set(xlabel='Date', ylabel='AdjClose',title='Stock price prediction Random Forest')
ax1.plot(use_index=True)
ax1.plot(x_test.index.values,y_test,color='r')
ax1.plot(x_test.index.values,y_pred,color='b')
plt.show
ax2.set(xlabel='Date', ylabel='AdjClose',title='Stock price prediction Regression')
ax2.plot(use_index=True)
ax2.plot(x_test.index.values,y_test,color='r')
ax2.plot(x_test.index.values,predict_y,color='b')
plt.show


ax3.set(xlabel='Date', ylabel='AdjClose',title='Stock price prediction KNN')
ax3.plot(use_index=True)
ax3.plot(x_test.index.values,y_test,color='r')
ax3.plot(x_test.index.values,predd,color='b')
plt.show


